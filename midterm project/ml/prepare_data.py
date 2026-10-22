"""Collect public corpora, retain provenance, add disclosed bilingual seed data.

Run: python -m ml.prepare_data
Never visit URLs found inside messages. Synthetic families and near duplicates
are grouped BEFORE splitting; evaluation must distinguish synthetic/public data.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

from ml.common import ROOT, TYPES, group_fingerprint, language, normalize
from ml.explain import evidence, weak_subtype

SEED = 20261023
PHISH_REVISION = '55caf9af01ae498d68ff44886ff69a4b165eb54b'
SOURCES = {
    'uci_sms.zip': {'url': 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip', 'license': 'CC BY 4.0 (UCI dataset page)', 'source': 'uci_sms'},
    'phishing_texts.json': {'url': f'https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/{PHISH_REVISION}/texts.json', 'license': 'Apache-2.0 declared by compilation; upstream licenses/provenance must also be respected', 'source': 'hf_phishing_texts'},
}

def download(raw):
    manifest = {}
    for filename, info in SOURCES.items():
        target = raw / filename
        if not target.exists():
            print('Downloading', filename, flush=True)
            req = urllib.request.Request(info['url'], headers={'User-Agent': 'MailGuard-academic-project/1.0'})
            with urllib.request.urlopen(req, timeout=120) as response:
                content = response.read(120_000_001)
            if len(content) > 120_000_000:
                raise ValueError('Unexpectedly large dataset response')
            tmp = target.with_suffix('.part')
            tmp.write_bytes(content)
            tmp.replace(target)
        content = target.read_bytes()
        manifest[filename] = {**info, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    return manifest

def row(text, label, source, annotation, subtype=None, group=None):
    text = normalize(text)
    return {'text': text, 'label': label, 'scam_type': subtype, 'language': language(text), 'source': source, 'annotation_method': annotation, 'group': group or group_fingerprint(text), 'id': hashlib.sha256((source + '\0' + text).encode()).hexdigest()[:20]}

def synthetic(variants):
    templates = json.loads((ROOT / 'data/seed_templates.json').read_text())
    rng = random.Random(SEED)
    records = []
    for category, pairs in templates.items():
        for i, pair in enumerate(pairs):
            for variant in range(variants):
                values = {'day': rng.choice(['Monday', 'Tuesday', 'Friday', '週一', '週三', '週五']), 'n': rng.randint(1000, 999999), 'amount': rng.choice([500, 1200, 3600, 8000, 12000, 30000, 68000]), 'rate': rng.choice([3, 5, 8, 12, 20, 30])}
                for lang, template in zip(['zh', 'en'], pair):
                    if lang == 'zh':
                        values['day'] = rng.choice(['週一', '週二', '週五', '下週三', '週末'])
                    else:
                        values['day'] = rng.choice(['Monday', 'Tuesday', 'Friday', 'next Wednesday', 'the weekend'])
                    label = category if category in ['normal', 'spam'] else 'scam'
                    records.append(row(template.format(**values), label, 'bilingual_templates', 'synthetic_template_label', category if category in TYPES else None, f'template:{category}:{i}'))
    return records

def public_records(raw, max_per_class):
    records, quarantine = [], []
    with zipfile.ZipFile(raw / 'uci_sms.zip') as zipped:
        sms = zipped.read('SMSSpamCollection').decode('utf-8')
    for i, line in enumerate(sms.splitlines()):
        if '\t' not in line:
            continue
        label, text = line.split('\t', 1)
        cues = evidence(text)
        # The UCI binary spam label does not distinguish scam from marketing.
        # Quarantine suspicious positives rather than invent scam ground truth.
        if label == 'spam' and any(c['scam_type'] for c in cues):
            quarantine.append({'source': 'uci_sms', 'source_row': i, 'text': text, 'reason': 'spam_may_be_scam_requires_review'})
            continue
        records.append(row(text, 'normal' if label == 'ham' else 'spam', 'uci_sms', 'source_binary_label'))
    obj = json.loads((raw / 'phishing_texts.json').read_text())
    if isinstance(obj, dict):
        if 'text' in obj and isinstance(obj['text'], list):
            obj = [dict(zip(obj, values)) for values in zip(*obj.values())]
        elif 'data' in obj:
            obj = obj['data']
        else:
            raise ValueError('Unknown phishing corpus structure')
    frame = pd.DataFrame(obj)
    for value in [0, 1]:
        selected = frame[frame['label'].astype(int) == value]
        selected = selected.sample(min(len(selected), max_per_class), random_state=SEED)
        for _, item in selected.iterrows():
            text = str(item['text'])
            subtype = weak_subtype(text) if value else None
            record = row(text, 'scam' if value else 'normal', 'hf_phishing_texts', 'source_phishing_label', subtype)
            record['subtype_annotation'] = 'weak_rule_label' if subtype else None
            records.append(record)
    # Prevent the normal SMS class from overwhelming all other sources.
    df = pd.DataFrame(records)
    chunks = []
    for _, group in df.groupby(['source', 'label']):
        chunks.append(group.sample(min(len(group), 3000), random_state=SEED))
    return pd.concat(chunks).to_dict('records'), quarantine

def cluster_near_duplicates(frame):
    """Union template families, normalized identities and cosine>=0.94 pairs."""
    parent = list(range(len(frame)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for key in ['group', 'fingerprint']:
        seen = {}
        for i, value in enumerate(frame[key]):
            if value in seen:
                union(i, seen[value])
            else:
                seen[value] = i
    matrix = HashingVectorizer(analyzer='char', ngram_range=(4, 4), n_features=2**18, alternate_sign=False, norm='l2').transform(frame['text'])
    count = 0
    for lo in range(0, len(frame), 128):
        similarities = cosine_similarity(matrix[lo:lo+128], matrix, dense_output=False).tocoo()
        selected = (similarities.data >= .94) & ((similarities.row+lo) < similarities.col)
        for a, b in zip(similarities.row[selected]+lo, similarities.col[selected]):
            union(int(a), int(b)); count += 1
    frame['group'] = [f'cluster:{find(i)}' for i in range(len(frame))]
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phishing-per-class', type=int, default=1600)
    parser.add_argument('--template-variants', type=int, default=20)
    args = parser.parse_args()
    raw, processed = ROOT / 'data/raw', ROOT / 'data/processed'
    raw.mkdir(parents=True, exist_ok=True); processed.mkdir(parents=True, exist_ok=True)
    manifest = download(raw)
    records, quarantine = public_records(raw, args.phishing_per_class)
    records += synthetic(args.template_variants)
    frame = pd.DataFrame(records)
    frame = frame[frame['text'].str.len().between(8, 12000)].copy()
    frame['fingerprint'] = frame['text'].map(group_fingerprint)
    conflicts = frame.groupby('fingerprint')['label'].nunique()
    bad = set(conflicts[conflicts > 1].index)
    removed_conflicts = len(frame[frame['fingerprint'].isin(bad)])
    frame = frame[~frame['fingerprint'].isin(bad)].drop_duplicates('text').reset_index(drop=True)
    print('Grouping', len(frame), 'records before splitting', flush=True)
    near_pairs = cluster_near_duplicates(frame)
    stratify = frame['label'] + ':' + frame['language']
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_val_idx, test_idx = next(outer.split(frame, stratify, frame['group']))
    train_val = frame.iloc[train_val_idx].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED+1)
    train_idx, val_idx = next(inner.split(train_val, train_val['label']+':'+train_val['language'], train_val['group']))
    splits = {'train': train_val.iloc[train_idx].copy(), 'validation': train_val.iloc[val_idx].copy(), 'test': frame.iloc[test_idx].copy()}
    summary = {}
    for name, subset in splits.items():
        assert set(subset['label']) == {'normal', 'spam', 'scam'}
        subset.to_json(processed / f'{name}.jsonl', orient='records', lines=True, force_ascii=False)
        summary[name] = {'n': len(subset), 'labels': subset['label'].value_counts().to_dict(), 'languages': subset['language'].value_counts().to_dict(), 'sources': subset['source'].value_counts().to_dict(), 'subtypes': subset['scam_type'].value_counts().to_dict(), 'sha256': hashlib.sha256((processed / f'{name}.jsonl').read_bytes()).hexdigest()}
    for a, b in [('train', 'test'), ('train', 'validation'), ('validation', 'test')]:
        assert not set(splits[a]['group']) & set(splits[b]['group'])
        assert not set(splits[a]['fingerprint']) & set(splits[b]['fingerprint'])
    # Keep a review queue; uncertain examples are never silently called gold labels.
    pd.DataFrame(quarantine).to_json(processed / 'review_queue.jsonl', orient='records', lines=True, force_ascii=False)
    audit = frame.groupby(['source', 'label'], group_keys=False).sample(n=8, random_state=SEED)
    audit.to_csv(processed / 'annotation_audit_sample.csv', index=False)
    result = {'seed': SEED, 'sources': manifest, 'splits': summary, 'quarantined_uci': len(quarantine), 'removed_conflicting_rows': removed_conflicts, 'near_duplicate_pairs_grouped': near_pairs, 'group_overlap': 0, 'preprocessing': ['NFKC', 'HTML unescape and tag removal', 'OpenCC s2twp simplified-to-traditional', 'zero-width removal', 'whitespace normalization'], 'label_policy': 'scam takes precedence over spam; normal is non-promotional benign content; spam is unsolicited marketing; source labels are imperfect', 'annotation_limits': ['Chinese and six-category supervision are primarily authored synthetic templates, not verified real-world scam records.', 'Public phishing subtypes are weak regex labels and must be evaluated separately.', 'Source binary labels may contain noise; non-phishing does not necessarily mean non-promotional.', 'Synthetic translations/variants stay in one split by template family.', 'No URLs in messages were visited.']}
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports/dataset_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
