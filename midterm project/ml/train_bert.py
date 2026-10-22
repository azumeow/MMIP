"""Fine-tune multilingual BERT; V100 full fine-tune or CPU partial fine-tune.

No Trainer/remote dataset code or external inference API is used.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from ml.common import ROOT, LABELS, TYPES
from ml.bert_model import PRETRAINED, PRETRAINED_REVISION, BertMultiTask
from ml.evaluate import report, save_report

class TextDataset(Dataset):
    def __init__(self, frame):
        self.rows = frame.to_dict('records')
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return self.rows[index]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=192)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--head-learning-rate', type=float, default=5e-4)
    parser.add_argument('--freeze-layers', type=int, default=0, help='0=full fine-tune; 11=CPU partial fine-tune (last encoder layer + heads)')
    parser.add_argument('--train-limit', type=int, default=0, help='0=all training examples; never limits held-out test data')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.freeze_layers <= 12:
        parser.error('--freeze-layers must be between 0 and 12')
    random.seed(20261023); np.random.seed(20261023); torch.manual_seed(20261023)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA was requested but is unavailable. Install a CUDA PyTorch build and verify nvidia-smi.')
    frames = {s: pd.read_json(ROOT / 'data/processed' / f'{s}.jsonl', lines=True) for s in ['train', 'validation', 'test']}
    original_train_n = len(frames['train'])
    if args.train_limit and args.train_limit < original_train_n:
        selected, _ = train_test_split(frames['train'], train_size=args.train_limit, stratify=frames['train']['label']+':'+frames['train']['language'], random_state=20261023)
        frames['train'] = selected
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, revision=PRETRAINED_REVISION)
    encoder = AutoModel.from_pretrained(PRETRAINED, revision=PRETRAINED_REVISION, use_safetensors=True)
    model = BertMultiTask(encoder.config, encoder).to(device)
    if args.freeze_layers:
        for parameter in model.encoder.embeddings.parameters():
            parameter.requires_grad = False
        for layer in model.encoder.encoder.layer[:args.freeze_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        # The unused NSP pooler is not part of the feature path.
        if model.encoder.pooler:
            for parameter in model.encoder.pooler.parameters():
                parameter.requires_grad = False

    def collate(rows):
        inputs = tokenizer([r['text'] for r in rows], truncation=True, padding=True, max_length=args.max_length, return_tensors='pt')
        labels = torch.tensor([LABELS.index(r['label']) for r in rows])
        subtypes = torch.tensor([TYPES.index(r['scam_type']) if r.get('scam_type') in TYPES else -100 for r in rows])
        return inputs, labels, subtypes

    loaders = {s: DataLoader(TextDataset(frame), batch_size=args.batch_size, shuffle=s=='train', collate_fn=collate, num_workers=0) for s, frame in frames.items()}
    counts = frames['train']['label'].value_counts()
    weights = torch.tensor([len(frames['train'])/(len(LABELS)*counts[k]) for k in LABELS], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': args.learning_rate},
        {'params': list(model.classifier.parameters()) + list(model.subtype_classifier.parameters()), 'lr': args.head_learning_rate},
    ], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    folder = ROOT / 'artifacts/bert'
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(folder); model.encoder.config.save_pretrained(folder)
    started = time.perf_counter()
    history, best = [], -1.

    @torch.inference_mode()
    def predict(loader):
        model.eval()
        logits, subtype_logits, gold = [], [], []
        for inputs, y, sub in loader:
            a, b = model(**{k:v.to(device) for k,v in inputs.items()})
            logits.append(a.cpu()); subtype_logits.append(b.cpu()); gold.append(y)
        return torch.cat(logits), torch.cat(subtype_logits), torch.cat(gold)

    for epoch in range(args.epochs):
        model.train()
        total, n = 0., 0
        for step, (inputs, labels, subtypes) in enumerate(loaders['train']):
            inputs = {k:v.to(device) for k,v in inputs.items()}
            labels, subtypes = labels.to(device), subtypes.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type=='cuda', dtype=torch.float16):
                main, subtype = model(**inputs)
                loss = criterion(main, labels)
                mask = subtypes != -100
                if mask.any():
                    loss = loss + .6 * nn.functional.cross_entropy(subtype[mask], subtypes[mask])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer); scaler.update()
            total += loss.item() * len(labels); n += len(labels)
            if step % 25 == 0:
                print(f'epoch={epoch+1}/{args.epochs} batch={step+1}/{len(loaders["train"])} loss={loss.item():.4f} elapsed={time.perf_counter()-started:.1f}s', flush=True)
        val_logits, _, val_gold = predict(loaders['validation'])
        score = f1_score(val_gold.numpy(), val_logits.argmax(1).numpy(), average='macro', zero_division=0)
        record = {'epoch': epoch+1, 'train_loss': total/n, 'validation_macro_f1': float(score)}
        history.append(record); print(json.dumps(record), flush=True)
        if score > best:
            best = score
            state = {k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
            save_file(state, str(folder/'model.safetensors'))
            (folder/'best_epoch.json').write_text(json.dumps(record))
        (ROOT/'reports/bert_history.json').write_text(json.dumps(history, indent=2))
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(folder/'model.safetensors')))
    # Temperature scaling is fitted ONLY on validation logits.
    val_logits, _, val_gold = predict(loaders['validation'])
    log_temperature = torch.zeros(1, requires_grad=True)
    calibration_optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=60)
    def closure():
        calibration_optimizer.zero_grad()
        loss = nn.functional.cross_entropy(val_logits/log_temperature.exp().clamp(.2, 10.), val_gold)
        loss.backward()
        return loss
    calibration_optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(.2, 10.).item())
    metadata = {'pretrained': PRETRAINED, 'revision': PRETRAINED_REVISION, 'architecture': 'multilingual BERT + 3-class head + 6-type head', 'labels': LABELS, 'subtypes': TYPES, 'max_length': args.max_length, 'temperature': temperature, 'calibration': 'temperature scaling on validation split', 'subtype_calibration': 'uncalibrated synthetic/weak supervision', 'device': str(device), 'frozen_encoder_layers': args.freeze_layers, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 'total_parameters': sum(p.numel() for p in model.parameters()), 'train_n': len(frames['train']), 'available_train_n': original_train_n, 'epochs': args.epochs, 'history': history, 'training_seconds': round(time.perf_counter()-started, 2)}
    test_logits, type_logits, _ = predict(loaders['test'])
    probabilities = (test_logits/temperature).softmax(1).numpy()
    result = report(frames['test'], probabilities, 'bert', metadata)
    test = frames['test'].reset_index(drop=True)
    for method in ['synthetic_template_label', 'source_phishing_label']:
        selected = test['scam_type'].isin(TYPES) & (test['annotation_method'] == method)
        if selected.any():
            y = test.loc[selected,'scam_type']
            pred = np.array(TYPES)[type_logits.numpy()[selected].argmax(1)]
            result.setdefault('subtype_evaluation', {})[method] = {'n':int(selected.sum()), 'report':classification_report(y,pred,labels=TYPES,output_dict=True,zero_division=0)}
    save_report(result, ROOT/'reports/bert_metrics.json')
    # This marker is written only after training and evaluation completed.
    (folder/'training_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print('BERT training, calibration, held-out evaluation and artifact saving completed', flush=True)

if __name__ == '__main__':
    main()
