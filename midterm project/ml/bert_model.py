"""A real multilingual BERT encoder with two supervised classification heads."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from ml.common import LABELS, TYPES

PRETRAINED = 'google-bert/bert-base-multilingual-cased'
PRETRAINED_REVISION = '3f076fdb1ab68d5b2880cb87a0886f315b8146f8'

class BertMultiTask(nn.Module):
    def __init__(self, config, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else AutoModel.from_config(config)
        self.dropout = nn.Dropout(.15)
        self.classifier = nn.Linear(config.hidden_size, len(LABELS))
        self.subtype_classifier = nn.Linear(config.hidden_size, len(TYPES))

    def forward(self, **inputs):
        encoded = self.encoder(**inputs)
        # Attention-mask-weighted mean preserves bilingual phrase information.
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (encoded.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        hidden = self.dropout(pooled)
        return self.classifier(hidden), self.subtype_classifier(hidden)

class BertPredictor:
    def __init__(self, folder: Path, device='cpu'):
        self.device = torch.device(device)
        self.metadata = json.loads((folder / 'training_metadata.json').read_text())
        self.max_length = self.metadata['max_length']
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        config = AutoConfig.from_pretrained(folder, local_files_only=True)
        self.model = BertMultiTask(config)
        self.model.load_state_dict(load_file(str(folder / 'model.safetensors')))
        self.model.to(self.device).eval()
        self.temperature = self.metadata.get('temperature', 1.)

    @torch.inference_mode()
    def predict_batch(self, texts):
        tokens = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        main, subtype = self.model(**tokens)
        return (main / self.temperature).softmax(-1).cpu().numpy(), subtype.softmax(-1).cpu().numpy()

    def predict(self, text):
        length = len(self.tokenizer(text, add_special_tokens=True, truncation=False)['input_ids'])
        probabilities, subtypes = self.predict_batch([text])
        return dict(zip(LABELS, map(float, probabilities[0]))), dict(zip(TYPES, map(float, subtypes[0]))), length > self.max_length
