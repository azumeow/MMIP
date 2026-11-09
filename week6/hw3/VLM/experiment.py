"""InternVL image-conditioned SFT: frozen vision + NF4 language base + LoRA.

Run from the repository root with ``python -m hw3.VLM.experiment train``.
Frozen image features are cached outside the submission; only assistant tokens
are supervised. Test data may be read by integrity checks, but never enters
the training forward pass, gradient updates, or checkpoint selection.
"""
import argparse
import contextlib
import importlib
import importlib.metadata
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from hw3.VLM.common import DATA, MODEL, MODEL_REVISION, ROOT, SEED, read_json, sha256, write_json

CONFIG = {
    'model': MODEL, 'revision': MODEL_REVISION, 'seed': SEED,
    'epochs': 3, 'micro_batch_size': 1, 'gradient_accumulation': 8,
    'learning_rate': 5e-5, 'warmup_ratio': .1, 'weight_decay': 0.0,
    'max_grad_norm': 1.0, 'lora_rank': 8, 'lora_alpha': 16,
    'lora_dropout': 0.0, 'lora_targets': ['q_proj', 'v_proj'],
    'quantization': 'NF4', 'double_quantization': False,
    'compute_dtype': 'float16', 'trainable_dtype': 'float32',
    'vision_dtype': 'float16', 'vision_and_projector': 'frozen',
    'max_dynamic_tiles': 2, 'thumbnail': True, 'tile_size': 448,
    'max_sequence_length': 1536, 'gradient_checkpointing': 'non-reentrant',
    'attention': 'eager', 'selection': 'lowest validation token-mean NLL',
    'max_new_tokens': 128, 'do_sample': False,
    'system_message': '你是一個圖片描述助手。',
}


def seed_all():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def autocast():
    return torch.autocast('cuda', dtype=torch.float16)


def finite(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'Nonfinite {label}; training stopped, not silently skipped')


def data_hashes():
    return {name: sha256(DATA / name) for name in
            ['train_sft_300.json', 'validation_60.json', 'target_test_60.json', 'sources.json']}


def pixels(path):
    """InternVL dynamic tiling, up to two tiles plus a global thumbnail."""
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    with Image.open(path) as source:
        image = source.convert('RGB')
    width, height = image.size
    size, maximum = CONFIG['tile_size'], CONFIG['max_dynamic_tiles']
    grids = sorted({(w, h) for w in range(1, maximum + 1)
                    for h in range(1, maximum + 1) if w * h <= maximum},
                   key=lambda g: (g[0] * g[1], g[0], g[1]))
    best, delta = (1, 1), float('inf')
    for grid in grids:
        diff = abs(width / height - grid[0] / grid[1])
        if diff < delta or (diff == delta and width * height > .5 * size * size * grid[0] * grid[1]):
            best, delta = grid, diff
    resized = image.resize((size * best[0], size * best[1]))
    tiles = [resized.crop((x * size, y * size, (x + 1) * size, (y + 1) * size))
             for y in range(best[1]) for x in range(best[0])]
    if len(tiles) > 1:
        tiles.append(image.resize((size, size)))
    transform = T.Compose([T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
    return torch.stack([transform(tile) for tile in tiles])


def load_model(cache_dir, train=False, adapter=None):
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    import bitsandbytes as bnb
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False,
        llm_int8_skip_modules=['vision_model', 'mlp1', 'lm_head'])
    model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION, cache_dir=cache_dir,
        trust_remote_code=True, use_flash_attn=False, torch_dtype=torch.float16,
        quantization_config=quant, device_map={'': 0}, low_cpu_mem_usage=True,
        local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION,
        cache_dir=cache_dir, trust_remote_code=True, use_fast=False, local_files_only=True)
    model.system_message = CONFIG['system_message']
    model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if train:
        # Prepare only the LM: keep frozen vision and projector in FP16.
        # Transformers marks the outer custom VLM as 4-bit, not its nested LM.
        # Propagate the flag so PEFT dispatches its actual Linear4bit adapters.
        model.language_model.is_loaded_in_4bit = True
        lm = prepare_model_for_kbit_training(model.language_model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False})
        lm = get_peft_model(lm, LoraConfig(task_type=TaskType.CAUSAL_LM,
            r=CONFIG['lora_rank'], lora_alpha=CONFIG['lora_alpha'],
            lora_dropout=CONFIG['lora_dropout'], target_modules=CONFIG['lora_targets']))
        # The nested Qwen config retains an upstream developer's local path;
        # the actual base weights come from this pinned InternVL repository.
        lm.peft_config['default'].base_model_name_or_path = MODEL
        lm.peft_config['default'].revision = MODEL_REVISION
        for parameter in lm.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        model.language_model = lm
    elif adapter is not None:
        model.language_model.is_loaded_in_4bit = True
        model.language_model = PeftModel.from_pretrained(model.language_model, adapter, is_trainable=False)
    lm = model.language_model
    lm.config.use_cache = False
    lm.generation_config.do_sample = False
    lm.generation_config.temperature = None
    lm.generation_config.top_p = None
    lm.generation_config.top_k = None
    quantized = [n for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    if not quantized or any(n.startswith(('vision_model.', 'mlp1.')) for n in quantized):
        raise RuntimeError('Incorrect quantization: expected only language linear layers')
    if train and not lm.is_gradient_checkpointing:
        raise RuntimeError('Expected language gradient checkpointing to be enabled')
    if any(p.requires_grad for module in [model.vision_model, model.mlp1] for p in module.parameters()):
        raise RuntimeError('Vision/projector must remain frozen')
    model.eval()
    return model, tokenizer, len(quantized)


def conversation(model):
    module = importlib.import_module(model.__class__.__module__)
    conv = module.get_conv_template(model.template)
    conv.system_message = model.system_message
    return conv


def encode(model, tokenizer, prompt, visual_tokens, answer=None):
    conv = conversation(model)
    conv.append_message(conv.roles[0], '<image>\n' + prompt)
    conv.append_message(conv.roles[1], None)
    prefix = conv.get_prompt().replace('<image>', '<img>' + '<IMG_CONTEXT>' * visual_tokens + '</img>', 1)
    prefix_ids = tokenizer(prefix, add_special_tokens=True)['input_ids']
    if sum(i == model.img_context_token_id for i in prefix_ids) != visual_tokens:
        raise ValueError('Image-token/feature alignment failed')
    if answer is None:
        return prefix_ids, len(prefix_ids), None
    ending = conv.sep.strip()
    # Assert the assistant boundary is not changed by subword token merging.
    full = tokenizer(prefix + answer + ending, add_special_tokens=True)['input_ids']
    if full[:len(prefix_ids)] != prefix_ids:
        raise ValueError('Assistant boundary is not a token prefix')
    if len(full) > CONFIG['max_sequence_length']:
        raise ValueError('Sequence too long; never silently truncate image or answer')
    if len(full) <= len(prefix_ids) or full[-1] != tokenizer.convert_tokens_to_ids(ending):
        raise ValueError('Missing supervised answer/EOS')
    return full, len(prefix_ids), full[len(prefix_ids):]


@torch.no_grad()
def feature(model, row, cache_dir):
    import hashlib
    key = hashlib.sha256((MODEL_REVISION + row['image_sha256'] +
        str((CONFIG['max_dynamic_tiles'], CONFIG['tile_size'], CONFIG['thumbnail'], 'imagenet'))).encode()).hexdigest()
    path = Path(cache_dir) / (key + '.pt')
    if path.exists():
        output = torch.load(path, map_location='cpu', weights_only=True)
    else:
        if sha256(DATA / row['image']) != row['image_sha256']:
            raise ValueError(f'Changed image: {row["id"]}')
        batch = pixels(DATA / row['image']).to('cuda', dtype=torch.float16)
        with autocast():
            output = model.extract_feature(batch).reshape(-1, model.language_model.config.hidden_size).detach().cpu()
        finite(output, 'image features')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
    if output.dtype != torch.float16 or output.ndim != 2 or output.shape[0] % model.num_image_token:
        raise ValueError('Invalid cached image features')
    return output


def prepare(model, tokenizer, rows, cache_dir):
    result = []
    for index, row in enumerate(rows, 1):
        visual = feature(model, row, cache_dir)
        ids, prefix, targets = encode(model, tokenizer, row['prompt'], len(visual), row['answer'])
        result.append({'id': row['id'], 'ids': ids, 'prefix': prefix,
                       'targets': targets, 'visual': visual})
        if index % 30 == 0:
            print(f'Prepared {index}/{len(rows)} image-conditioned examples', flush=True)
    return result


def embeddings(model, ids, visual, training=False):
    ids = torch.tensor([ids], dtype=torch.long, device='cuda')
    embedded = model.language_model.get_input_embeddings()(ids).clone()
    selected = ids == model.img_context_token_id
    if int(selected.sum()) != len(visual):
        raise ValueError('Image feature count mismatch')
    embedded[selected] = visual.to(device='cuda', dtype=embedded.dtype)
    if training:
        embedded.requires_grad_(True)
    return embedded, torch.ones_like(ids)


def example_loss(model, item, training=False):
    embedded, mask = embeddings(model, item['ids'], item['visual'], training)
    # Select only positions predicting assistant tokens, avoiding a full
    # [image+prompt+answer, vocabulary] logits tensor. This is ordinary SFT CE.
    positions = torch.arange(item['prefix'] - 1, len(item['ids']) - 1, device='cuda')
    with autocast():
        outputs = model.language_model(inputs_embeds=embedded, attention_mask=mask,
            logits_to_keep=positions, use_cache=False)
    logits = outputs.logits[0].float()
    targets = torch.tensor(item['targets'], dtype=torch.long, device='cuda')
    if logits.shape[0] != len(targets):
        raise ValueError('Loss position/target mismatch')
    loss = F.cross_entropy(logits, targets)
    finite(loss, 'SFT loss')
    return loss


@torch.no_grad()
def validation(model, items):
    model.eval()
    total, tokens, scores = 0., 0, []
    for item in items:
        loss = example_loss(model, item).item()
        n = len(item['targets'])
        total += loss * n
        tokens += n
        scores.append({'id': item['id'], 'nll': loss, 'supervised_tokens': n})
    return {'token_mean_nll': total / tokens, 'perplexity': math.exp(total / tokens),
            'supervised_tokens': tokens, 'count': len(items), 'examples': scores}


def train(args):
    from hw3.VLM.common import validate_data
    adapter = ROOT / 'checkpoints/sft_adapter'
    record_path = ROOT / 'results/sft_training.json'
    if adapter.exists() or record_path.exists():
        raise FileExistsError('Existing training output: preserve it before starting another experiment')
    seed_all()
    checked = validate_data()
    model, tokenizer, quantized = load_model(args.model_cache, train=True)
    train_rows = read_json(DATA / 'train_sft_300.json')['examples']
    val_rows = read_json(DATA / 'validation_60.json')['examples']
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    train_items = prepare(model, tokenizer, train_rows, args.feature_cache)
    val_items = prepare(model, tokenizer, val_rows, args.feature_cache)
    initial_validation = validation(model, val_items)
    print(f'Initial validation NLL: {initial_validation["token_mean_nll"]:.6f}', flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not named or any('lora_' not in name for name in named):
        raise RuntimeError('Only LoRA parameters should be trainable')
    initial = {n: p.detach().cpu().clone() for n, p in named.items()}
    updates_per_epoch = math.ceil(len(train_items) / CONFIG['gradient_accumulation'])
    total_steps = updates_per_epoch * CONFIG['epochs']
    warmup = max(1, math.ceil(total_steps * CONFIG['warmup_ratio']))
    optimizer = torch.optim.AdamW(params, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    def factor(step):
        return min((step + 1) / warmup, max(0., (total_steps - step) / max(1, total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.)
    best, best_epoch, history, epochs = float('inf'), None, [], []
    torch.cuda.reset_peak_memory_stats()
    record = {'status': 'running', 'started_at': started_at,
        'config': CONFIG, 'data_hashes': data_hashes(), 'dataset_validation': checked,
        'environment': {n: importlib.metadata.version(n) for n in
            ['torch', 'transformers', 'peft', 'bitsandbytes', 'huggingface-hub', 'accelerate', 'torchvision']},
        'gpu': torch.cuda.get_device_name(), 'quantized_linear_layers': quantized,
        'trainable_parameters': sum(p.numel() for p in params),
        'trainable_parameter_names': list(named), 'initial_validation': initial_validation,
        'sequence_lengths': {'minimum': min(len(x['ids']) for x in train_items),
                             'maximum': max(len(x['ids']) for x in train_items)},
        'training_ids': [x['id'] for x in train_items],
        'test_used_for_training_or_selection': False, 'updates': history, 'epochs': epochs}
    write_json(record_path, record)
    for epoch in range(1, CONFIG['epochs'] + 1):
        order = list(train_items)
        random.Random(SEED + epoch).shuffle(order)
        model.language_model.train()
        epoch_losses = []
        for start in range(0, len(order), CONFIG['gradient_accumulation']):
            batch = order[start:start + CONFIG['gradient_accumulation']]
            optimizer.zero_grad(set_to_none=True)
            values = []
            for item in batch:
                loss = example_loss(model, item, training=True)
                values.append(loss.item())
                scaler.scale(loss / len(batch)).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, CONFIG['max_grad_norm'])
            finite(norm, 'LoRA gradient norm')
            lr = optimizer.param_groups[0]['lr']
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_losses.extend(values)
            history.append({'step': len(history) + 1, 'epoch': epoch,
                'ids': [x['id'] for x in batch], 'mean_answer_nll': sum(values) / len(values),
                'grad_norm': norm.item(), 'learning_rate': lr, 'loss_scale': scaler.get_scale()})
            record['elapsed_seconds'] = time.perf_counter() - started
            record['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
            record['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
            write_json(record_path, record)
            print(f'SFT epoch={epoch}/{CONFIG["epochs"]} update={len(history)}/{total_steps} '
                  f'nll={sum(values)/len(values):.4f} grad={norm.item():.4f} '
                  f'elapsed={record["elapsed_seconds"]:.1f}s', flush=True)
        val = validation(model, val_items)
        epochs.append({'epoch': epoch, 'train_example_mean_nll': sum(epoch_losses) / len(epoch_losses),
                       'validation': val})
        if val['token_mean_nll'] < best:
            best, best_epoch = val['token_mean_nll'], epoch
            model.language_model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
            tokenizer.save_pretrained(adapter)
            write_json(adapter / 'training_manifest.json', {'config': CONFIG,
                'data_hashes': data_hashes(), 'epoch': epoch, 'validation_nll': best,
                'loading': 'Load pinned InternVL NF4 base; attach this adapter to model.language_model, not the full VLM.'})
        record.update(best_epoch=best_epoch, best_validation_nll=best)
        write_json(record_path, record)
        print(f'Validation epoch={epoch} NLL={val["token_mean_nll"]:.6f}; best epoch={best_epoch}', flush=True)
    change = math.sqrt(sum((p.detach().float().cpu() - initial[n]).square().sum().item() for n, p in named.items()))
    if not change > 0 or len(history) != total_steps:
        raise RuntimeError('Training did not make the expected real parameter updates')
    record.update(status='completed', completed_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter() - started, optimizer_steps=len(history),
                  final_adapter_delta_l2=change, selected_adapter=str(adapter.relative_to(ROOT)),
                  selected_adapter_sha256=sha256(adapter / 'adapter_model.safetensors'))
    write_json(record_path, record)
    print(f'TRAINING_COMPLETED best_epoch={best_epoch} adapter={adapter}', flush=True)


@torch.no_grad()
def generate(model, tokenizer, row, visual):
    ids, _, _ = encode(model, tokenizer, row['prompt'], len(visual))
    embedded, mask = embeddings(model, ids, visual)
    eos = tokenizer.convert_tokens_to_ids(conversation(model).sep.strip())
    with autocast():
        output = model.language_model.generate(inputs_embeds=embedded, attention_mask=mask,
            max_new_tokens=CONFIG['max_new_tokens'], do_sample=False, temperature=None,
            top_p=None, top_k=None, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
            use_cache=True)
    generated = output[0].tolist()
    return {'answer': tokenizer.decode(generated, skip_special_tokens=True).strip(),
            'generated_tokens': len(generated), 'ended_with_eos': bool(generated and generated[-1] == eos),
            'input_tokens': len(ids), 'visual_tokens': len(visual)}


def evaluate(args):
    record = read_json(ROOT / 'results/sft_training.json')
    if record['status'] != 'completed' or record['data_hashes'] != data_hashes():
        raise ValueError('Complete training with unchanged data before test evaluation')
    paths = [ROOT / f'results/{name}.json' for name in ['baseline', 'after_sft']]
    if any(p.exists() for p in paths):
        raise FileExistsError('Evaluation outputs already exist; preserve them before rerunning')
    seed_all()
    model, tokenizer, _ = load_model(args.model_cache, adapter=ROOT / 'checkpoints/sft_adapter')
    rows = read_json(DATA / 'target_test_60.json')['examples']
    # Validate reloading and the cached-feature path against the model author's
    # actual image -> chat API after all training/selection decisions are fixed.
    first = rows[0]
    first_visual = feature(model, first, args.feature_cache)
    cached_answer = generate(model, tokenizer, first, first_visual)['answer']
    with torch.no_grad(), autocast():
        official_answer = model.chat(tokenizer,
            pixels(DATA / first['image']).to('cuda', dtype=torch.float16),
            '<image>\n' + first['prompt'],
            {'max_new_tokens': CONFIG['max_new_tokens'], 'do_sample': False,
             'temperature': None, 'top_p': None, 'top_k': None}, history=None).strip()
    if cached_answer != official_answer:
        raise RuntimeError('Cached-feature generation differs from the official image chat API')
    parity = {'id': first['id'], 'same_answer': True,
              'check': 'reloaded selected adapter: cached-feature generation == official pixel_values chat'}
    print('Official image-chat parity check passed', flush=True)
    for name, path in zip(['baseline', 'after_sft'], paths):
        results = []
        output = {'status': 'running', 'model': MODEL, 'revision': MODEL_REVISION,
            'adapter': None if name == 'baseline' else 'checkpoints/sft_adapter',
            'config': CONFIG, 'data_hashes': data_hashes(), 'image_chat_parity': parity, 'examples': results}
        started = time.perf_counter()
        ctx = model.language_model.disable_adapter() if name == 'baseline' else contextlib.nullcontext()
        with ctx:
            for row in rows:
                visual = feature(model, row, args.feature_cache)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                generated = generate(model, tokenizer, row, visual)
                torch.cuda.synchronize()
                results.append({'id': row['id'], 'image': row['image'], 'prompt': row['prompt'],
                    'reference_answer': row['answer'], 'reference_answers': row['reference_answers'],
                    'image_sha256': row['image_sha256'], **generated,
                    'generation_seconds': time.perf_counter() - tick})
                write_json(path, output)
                if len(results) % 10 == 0:
                    print(f'{name} {len(results)}/60', flush=True)
        output.update(status='completed', count=len(results), elapsed_seconds=time.perf_counter()-started)
        write_json(path, output)
    print('EVALUATION_COMPLETED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train', 'evaluate'])
    parser.add_argument('--model-cache', default=None, help='Hugging Face cache with pinned model already downloaded')
    parser.add_argument('--feature-cache', default='/tmp/hw3_vlm_features')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    (train if args.stage == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
