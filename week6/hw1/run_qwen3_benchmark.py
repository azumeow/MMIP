"""Qwen3-8B benchmark: FP16 vs bitsandbytes 4-bit on a single NVIDIA GPU.

Usage:
  python run_qwen3_benchmark.py --benchmark benchmark.json --mode both
  python run_qwen3_benchmark.py --benchmark benchmark.json --mode fp16
  python run_qwen3_benchmark.py --benchmark benchmark.json --mode 4bit

Files: result_fp16.json and result_4_bit.json (saved after every question).
"""

import argparse
import copy
import gc
import json
import os
import platform
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

try:
    import pynvml  # installed by nvidia-ml-py
except ImportError as exc:
    raise SystemExit('Missing GPU measurement package: pip install nvidia-ml-py') from exc

MODEL_NAME = 'Qwen/Qwen3-8B'


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def save_json_atomic(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temp, path)


class GPUMemorySampler:
    """Periodically sample total GPU memory USED by all processes via NVML."""

    def __init__(self, device_index=0, interval_seconds=0.1):
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval = interval_seconds
        self.samples_mib = []
        self.stop_event = threading.Event()
        self.thread = None

    def _read_mib(self):
        return pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / (1024 ** 2)

    def _run(self):
        while not self.stop_event.is_set():
            self.samples_mib.append(self._read_mib())
            self.stop_event.wait(self.interval)

    def start(self):
        self.samples_mib = [self._read_mib()]
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.samples_mib.append(self._read_mib())
        return {
            'mean_gpu_memory_mib': round(statistics.mean(self.samples_mib), 2),
            'peak_gpu_memory_mib': round(max(self.samples_mib), 2),
            'min_gpu_memory_mib': round(min(self.samples_mib), 2),
            'sample_count': len(self.samples_mib),
        }


def load_model(mode):
    kwargs = {
        'device_map': {'': 0},  # entire model on GPU 0; no CPU offload
        'torch_dtype': torch.float16,  # V100: FP16 compute, not BF16
        'low_cpu_mem_usage': True,
    }
    if mode == '4bit':
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    model.eval()
    return model


def run_mode(mode, benchmark, tokenizer, args, gpu_name, gpu_total_mib):
    filename = 'result_fp16.json' if mode == 'fp16' else 'result_4_bit.json'
    out_path = Path(args.output_dir) / filename
    print(f'\n=== Loading {MODEL_NAME}: {mode} ===', flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError as e:
            print(f"Warning: 無法重設顯存統計: {e}")
    load_start = time.perf_counter()
    model = load_model(mode)
    torch.cuda.synchronize(0)
    loading_seconds = time.perf_counter() - load_start

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    initial_used_mib = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2

    results = {
        'benchmark_name': benchmark.get('benchmark_name'),
        'benchmark_version': benchmark.get('version'),
        'model_name': MODEL_NAME,
        'precision': 'FP16' if mode == 'fp16' else '4-bit NF4 (FP16 compute)',
        'quantization': None if mode == 'fp16' else {
            'library': 'bitsandbytes', 'method': 'NF4',
            'compute_dtype': 'float16', 'double_quant': False,
        },
        'timestamp_utc': utc_now(),
        'status': 'running',
        'environment': {
            'gpu': gpu_name,
            'gpu_total_memory_mib': round(gpu_total_mib, 2),
            'pytorch_version': torch.__version__,
            'transformers_version': __import__('transformers').__version__,
            'cuda_version': torch.version.cuda,
            'python_version': platform.python_version(),
            'device_index': 0,
        },
        'generation_config': {
            'input_field': 'question',
            'thinking_enabled': False,
            'max_new_tokens': args.max_new_tokens,
            'do_sample': False,
            'temperature': None, 'top_p': None, 'top_k': None,
            'batch_size': 1,
            'chat_template': 'Qwen3 apply_chat_template',
            'system_prompt': None,
        },
        'memory_method': {
            'source': 'NVIDIA NVML device total used memory (not process-only)',
            'sampling_interval_ms': round(args.sample_interval * 1000),
            'sampled_scope': 'model.generate() per question, including input prefill and token decoding',
            'unit': 'MiB (1024**2 bytes)',
            'notes': 'Includes other processes on GPU. Compare on an otherwise idle GPU. '
                     'Excludes model loading. Mean is computed from periodic samples.'
        },
        'model_load_seconds': round(loading_seconds, 3),
        'model_loaded_gpu_memory_mib': round(initial_used_mib, 2),
        'total_questions': len(benchmark['questions']),
        'questions': [],
        'summary': {},
    }
    save_json_atomic(out_path, results)
    sampler = GPUMemorySampler(0, args.sample_interval)

    # No answer/reference_solution/evaluation goes into the model input.
    for i, item in enumerate(benchmark['questions'], 1):
        prompt = item['question']
        chat = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        encoded = tokenizer(chat, return_tensors='pt').to('cuda:0')
        input_tokens = encoded['input_ids'].shape[-1]
        torch.cuda.synchronize(0)
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError as e:
            print(f"Warning: 無法重設顯存統計: {e}")
        sampler.start()
        started = time.perf_counter()
        output = None
        error = None
        try:
            # Fresh explicit generation config avoids sampling flags warning.
            gen_config = GenerationConfig(
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            with torch.inference_mode():
                output = model.generate(**encoded, generation_config=gen_config)
            torch.cuda.synchronize(0)
        except Exception as exc:
            torch.cuda.synchronize(0)
            error = f'{type(exc).__name__}: {exc}'
        finally:
            elapsed = time.perf_counter() - started
            memory = sampler.stop()
            torch_peak_allocated_mib = torch.cuda.max_memory_allocated(0) / 1024**2

        if output is not None:
            generated_ids = output[0][input_tokens:]
            output_tokens = int(generated_ids.numel())
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            eos_ids = model.generation_config.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
            stopped_by_eos = bool(output_tokens and int(generated_ids[-1]) in eos_ids)
            truncated = (output_tokens >= args.max_new_tokens) and not stopped_by_eos
        else:
            output_tokens = 0
            response = None
            stopped_by_eos = False
            truncated = False

        result = {
            'id': item['id'],
            'category': item.get('category'),
            'question': prompt,
            'reference_answer': item.get('answer'),
            'generated_answer': response,
            'input_tokens': int(input_tokens),
            'generated_tokens': output_tokens,
            'total_tokens': int(input_tokens) + output_tokens,
            'inference_seconds': round(elapsed, 3),
            'output_tokens_per_second': round(output_tokens / elapsed, 3) if elapsed else None,
            'gpu_memory': memory,
            'torch_peak_allocated_mib': round(torch_peak_allocated_mib, 2),
            'stopped_by_eos': stopped_by_eos,
            'hit_max_new_tokens': truncated,
            'error': error,
            'answer_correct': None,  # not graded here
        }
        results['questions'].append(result)
        good = [q for q in results['questions'] if q['error'] is None]
        if good:
            results['summary'] = {
                'completed_questions': len(good),
                'error_questions': len(results['questions']) - len(good),
                'mean_gpu_memory_mib': round(statistics.mean(q['gpu_memory']['mean_gpu_memory_mib'] for q in good), 2),
                'max_gpu_memory_mib': round(max(q['gpu_memory']['peak_gpu_memory_mib'] for q in good), 2),
                'mean_inference_seconds': round(statistics.mean(q['inference_seconds'] for q in good), 3),
                'total_inference_seconds': round(sum(q['inference_seconds'] for q in good), 3),
                'total_generated_tokens': sum(q['generated_tokens'] for q in good),
                'overall_output_tokens_per_second': round(
                    sum(q['generated_tokens'] for q in good) / sum(q['inference_seconds'] for q in good), 3
                ),
                'hit_max_tokens_count': sum(q['hit_max_new_tokens'] for q in good),
                'mean_gpu_memory_definition': 'arithmetic mean of the per-question NVML memory means',
            }
        save_json_atomic(out_path, results)
        print(f'[{i:02d}/{len(benchmark["questions"])}] {item["id"]} '
              f'tokens={output_tokens} time={elapsed:.2f}s '
              f'GPU(mean/peak)={memory["mean_gpu_memory_mib"]:.0f}/'
              f'{memory["peak_gpu_memory_mib"]:.0f} MiB '
              f'{"ERROR: " + error if error else ""}', flush=True)
        del encoded, output
        # Clear per-question cached intermediates without unloading model.
        torch.cuda.empty_cache()

    results['status'] = 'completed' if all(q['error'] is None for q in results['questions']) else 'completed_with_errors'
    results['finished_at_utc'] = utc_now()
    save_json_atomic(out_path, results)
    print('Saved:', out_path.resolve(), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='benchmark.json')
    ap.add_argument('--mode', choices=['fp16', '4bit', 'both'], default='both')
    ap.add_argument('--output-dir', default='.')
    ap.add_argument('--max-new-tokens', type=int, default=4096)
    ap.add_argument('--sample-interval', type=float, default=0.1)
    args = ap.parse_args()
    if not 0.02 <= args.sample_interval <= 5:
        ap.error('--sample-interval must be between 0.02 and 5 seconds')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required to run this benchmark.')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(args.benchmark, encoding='utf-8') as f:
        benchmark = json.load(f)
    if not benchmark.get('questions'):
        raise ValueError('benchmark must contain nonempty questions list')
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8')
        gpu_total_mib = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**2
        print(f'GPU: {gpu_name} ({gpu_total_mib:.0f} MiB) | Questions: {len(benchmark["questions"])}')
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        modes = ['fp16', '4bit'] if args.mode == 'both' else [args.mode]
        for mode in modes:
            run_mode(mode, benchmark, tokenizer, args, gpu_name, gpu_total_mib)
    finally:
        pynvml.nvmlShutdown()


if __name__ == '__main__':
    main()
