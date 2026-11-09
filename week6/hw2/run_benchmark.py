"""Run either requested VLM on the same 20 images and verbatim task prompts.

Run one model per process to release all GPU state between models.
"""
import argparse
import importlib.metadata
import json
import platform
import random
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from common import DEFAULT_DATASET, MODELS, ROOT, SYSTEM_PROMPT, load_dataset, sha256, write_json


class MemorySampler:
    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.stop_event = threading.Event()
        self.values = []

    def sample(self):
        self.values.append(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1024**2)

    def start(self):
        self.values = []
        self.stop_event.clear()
        self.sample()
        def loop():
            while not self.stop_event.wait(0.1):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        return {'mean_mib': statistics.mean(self.values), 'peak_mib': max(self.values),
                'samples': len(self.values)}


def internvl_pixels(image, max_tiles):
    """Official dynamic tiling algorithm; deterministic ordering for tied grids."""
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    width, height = image.size
    size = 448
    grids = sorted({(w, h) for w in range(1, max_tiles+1)
                    for h in range(1, max_tiles+1) if w*h <= max_tiles},
                   key=lambda g: (g[0]*g[1], g[0], g[1]))
    best, delta = (1, 1), float('inf')
    for grid in grids:
        diff = abs(width/height - grid[0]/grid[1])
        if diff < delta or (diff == delta and width*height > .5*size*size*grid[0]*grid[1]):
            best, delta = grid, diff
    resized = image.resize((size*best[0], size*best[1]))
    tiles = [resized.crop((x*size, y*size, (x+1)*size, (y+1)*size))
             for y in range(best[1]) for x in range(best[0])]
    if len(tiles) > 1:
        tiles.append(image.resize((size, size)))
    transform = T.Compose([T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
    return torch.stack([transform(tile) for tile in tiles])


def load_adapter(args):
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
    model_id, revision = MODELS[args.model]
    # Preserve vision towers/projectors in FP16; quantize language linear layers.
    skip = ['vision_model', 'mlp1', 'visual', 'lm_head']
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=False, llm_int8_skip_modules=skip)
    kwargs = dict(revision=revision, torch_dtype=torch.float16,
                  quantization_config=quant, device_map={'': 0}, low_cpu_mem_usage=True,
                  local_files_only=args.offline, cache_dir=args.cache_dir)
    if args.model == 'internvl':
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                          use_flash_attn=False, **kwargs).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                                  trust_remote_code=True, use_fast=False,
                                                  local_files_only=args.offline, cache_dir=args.cache_dir)
        model.system_message = SYSTEM_PROMPT
        lm = model.language_model
        # Explicitly replace inherited sampling defaults, not just CLI flags.
        lm.generation_config.do_sample = False
        lm.generation_config.temperature = None
        lm.generation_config.top_p = None
        lm.generation_config.top_k = None
        captured = {}
        original_generate = lm.generate
        def capture_generate(*a, **kw):
            captured['input_tokens'] = int(kw['inputs_embeds'].shape[1])
            output = original_generate(*a, **kw)
            captured['ids'] = output[0].detach()
            return output
        lm.generate = capture_generate
        def prepare(image, prompt):
            pixels = internvl_pixels(image, args.internvl_max_tiles).to('cuda', dtype=torch.float16)
            return pixels, {'tiles': int(pixels.shape[0]), 'visual_tokens': int(pixels.shape[0])*model.num_image_token}
        def infer(pixels, prompt):
            answer = model.chat(tokenizer, pixels, '<image>\n'+prompt,
                                {'max_new_tokens': args.max_new_tokens, 'do_sample': False,
                                 'temperature': None, 'top_p': None, 'top_k': None}, history=None)
            ids = captured['ids']
            return answer, ids, captured['input_tokens']
        eos = [tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())]
    else:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, attn_implementation='eager', **kwargs).eval()
        processor = AutoProcessor.from_pretrained(model_id, revision=revision,
                                                  local_files_only=args.offline, cache_dir=args.cache_dir)
        processor.image_processor.size = {'shortest_edge': 256*32*32,
                                          'longest_edge': args.qwen_max_pixels}
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        def prepare(image, prompt):
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors='pt').to('cuda')
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
            grid = inputs['image_grid_thw'][0].tolist()
            return inputs, {'image_grid_thw': grid, 'visual_tokens': int(grid[0]*grid[1]*grid[2]//4)}
        def infer(inputs, prompt):
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, temperature=None, top_p=None, top_k=None)
            count = int(inputs['input_ids'].shape[1])
            ids = output[0, count:]
            return processor.tokenizer.decode(ids, skip_special_tokens=True).strip(), ids, count
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
    import bitsandbytes as bnb
    quantized = [name for name, layer in model.named_modules() if isinstance(layer, bnb.nn.Linear4bit)]
    if not quantized:
        raise RuntimeError('No 4-bit linear layers found')
    if any(name.startswith(('vision_model.', 'mlp1.', 'model.visual.', 'visual.')) for name in quantized):
        raise RuntimeError('Vision tower/projector was unexpectedly quantized')
    return model, prepare, infer, eos, len(quantized)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, choices=MODELS)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--internvl-max-tiles', type=int, default=6)
    ap.add_argument('--qwen-max-pixels', type=int, default=1280*32*32)
    ap.add_argument('--limit', type=int, help='Smoke test only; do not compare incomplete runs')
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--cache-dir', type=str, help='Optional Hugging Face model cache, outside the submission')
    args = ap.parse_args()
    if args.max_new_tokens < 1 or args.internvl_max_tiles < 1 or args.qwen_max_pixels < 256*32*32:
        ap.error('Invalid token or image budget')
    if args.limit is not None and not 1 <= args.limit <= 20:
        ap.error('--limit must be between 1 and 20')
    path = args.output or ROOT/f'results/{args.model}.json'
    if path.exists():
        raise SystemExit(f'Refusing to overwrite existing results: {path}; choose another --output')
    data = load_dataset(args.dataset)
    import numpy as np
    import torch
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sampler = MemorySampler()
    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'torchvision', 'transformers', 'accelerate', 'bitsandbytes', 'timm', 'einops', 'pillow']}
    result = {'status': 'loading', 'model': MODELS[args.model][0], 'revision': MODELS[args.model][1],
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'dataset_sha256': sha256(args.dataset), 'system_prompt': SYSTEM_PROMPT,
              'environment': {'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
                              'cuda': torch.version.cuda, 'packages': versions},
              'settings': {'precision': 'language NF4 4-bit, vision/projector FP16, compute FP16',
                           'double_quant': False, 'do_sample': False, 'seed': 42,
                           'max_new_tokens': args.max_new_tokens, 'batch_size': 1,
                           'attention': 'eager', 'internvl_max_tiles': args.internvl_max_tiles,
                           'qwen_max_pixels': args.qwen_max_pixels,
                           'memory_scope': 'NVML whole GPU, sampled every 100 ms',
                           'time_scope': 'adapter inference including vision encoder and generation; excludes CPU image preprocessing'},
              'questions': []}
    write_json(path, result)
    start = time.perf_counter()
    try:
        model, prepare, infer, eos, quantized_count = load_adapter(args)
        torch.cuda.synchronize()
        result['load_seconds'] = time.perf_counter()-start
        result['quantized_linear_layers'] = quantized_count
        sampler.sample()
        result['loaded_gpu_memory_mib'] = sampler.values[-1]
        result['status'] = 'running'
        write_json(path, result)
        for q in data['questions'][:args.limit]:
            with Image.open(args.dataset.resolve().parent/q['image_path']) as source:
                image = source.convert('RGB')
            prepared, visual = prepare(image, q['prompt'])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            sampler.start()
            start = time.perf_counter()
            error = None
            try:
                with torch.inference_mode():
                    answer, ids, input_count = infer(prepared, q['prompt'])
                torch.cuda.synchronize()
                output_count = int(ids.numel())
                truncated = output_count >= args.max_new_tokens and int(ids[-1]) not in (eos or [])
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                answer, output_count, input_count, truncated = None, 0, None, False
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            memory = sampler.stop()
            record = {'id': q['id'], 'prompt': q['prompt'], 'image_sha256': q['image_sha256'],
                      'answer': answer, 'input_tokens': input_count, 'output_tokens': output_count,
                      'seconds': elapsed, 'tokens_per_second': output_count/elapsed,
                      'visual_input': visual, 'gpu_memory': memory,
                      'torch_peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
                      'truncated': truncated, 'error': error}
            result['questions'].append(record)
            write_json(path, result)
            print(f'{args.model} {q["id"]}: {elapsed:.2f}s, {output_count} tokens, answer={answer!r}, error={error}', flush=True)
            del prepared
            torch.cuda.empty_cache()
        rows = result['questions']
        result['status'] = ('completed' if len(rows)==20 else 'smoke_test_completed') if all(r['error'] is None for r in rows) else 'completed_with_errors'
        result['summary'] = {'questions': len(rows), 'errors': sum(r['error'] is not None for r in rows),
                             'truncated': sum(r['truncated'] for r in rows),
                             'total_seconds': sum(r['seconds'] for r in rows),
                             'mean_gpu_memory_mib': statistics.mean(r['gpu_memory']['mean_mib'] for r in rows),
                             'peak_gpu_memory_mib': max(r['gpu_memory']['peak_mib'] for r in rows),
                             'total_output_tokens': sum(r['output_tokens'] for r in rows)}
        result['summary']['overall_tokens_per_second'] = result['summary']['total_output_tokens']/result['summary']['total_seconds']
        result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        result['status'] = 'failed'
        result['fatal_error'] = f'{type(exc).__name__}: {exc}'
        write_json(path, result)
        raise
    finally:
        sampler.nvml.nvmlShutdown()
    write_json(path, result)


if __name__ == '__main__':
    main()
