import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, f1_score
from ml.common import LABELS

def report(frame, probabilities, model_name, metadata=None):
    probs = np.asarray(probabilities)
    pred = np.array(LABELS)[probs.argmax(axis=1)]
    gold = frame['label'].to_numpy()
    confidence = probs.max(axis=1)
    correct = pred == gold
    ece = 0.0
    for lo in np.arange(0, 1, .1):
        selected = (confidence > lo) & (confidence <= lo + .1 + 1e-9)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    groups = {}
    for field in ['language', 'source', 'annotation_method']:
        groups[field] = {}
        for value in sorted(frame[field].unique()):
            selected = (frame[field] == value).to_numpy()
            present = sorted(set(gold[selected]))
            groups[field][value] = {'n': int(selected.sum()), 'present_labels': present, 'accuracy': float(accuracy_score(gold[selected], pred[selected])), 'macro_f1': float(f1_score(gold[selected], pred[selected], labels=LABELS, average='macro', zero_division=0)), 'macro_f1_present_labels': float(f1_score(gold[selected], pred[selected], labels=present, average='macro', zero_division=0))}
    # sklearn log_loss uses alphabetical class order even when labels are passed.
    alphabetical = sorted(LABELS)
    ordered = probs[:, [LABELS.index(label) for label in alphabetical]]
    return {'model': model_name, 'test_n': len(frame), 'labels': LABELS, 'accuracy': float(accuracy_score(gold, pred)), 'macro_f1': float(f1_score(gold, pred, labels=LABELS, average='macro', zero_division=0)), 'log_loss': float(log_loss(gold, ordered, labels=alphabetical)), 'ece_10_bins': float(ece), 'classification_report': classification_report(gold, pred, labels=LABELS, output_dict=True, zero_division=0), 'confusion_matrix': confusion_matrix(gold, pred, labels=LABELS).tolist(), 'groups': groups, 'metadata': metadata or {}, 'limitations': ['Evaluation includes synthetic examples; this is not a real-world fraud detection accuracy claim.', 'Source, language, and annotation-method slices are reported separately. Source macro-F1 includes all three classes; present-label macro-F1 is also provided.', 'Highest probability is model confidence, not a guarantee of safety or fraud.']}

def save_report(result, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"{result['model']}: test n={result['test_n']}, accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}", flush=True)
