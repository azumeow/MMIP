import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from ml.common import ROOT, LABELS, TYPES
from ml.evaluate import report, save_report

def pipeline():
    return Pipeline([
        ('features', FeatureUnion([
            ('chars', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2, max_features=70000, sublinear_tf=True)),
            ('words', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)),
        ])),
        ('lr', LogisticRegression(max_iter=700, class_weight='balanced', C=3.0, solver='lbfgs')),
    ])

def main():
    start = time.perf_counter()
    train, val, test = [pd.read_json(ROOT / 'data/processed' / f'{s}.jsonl', lines=True) for s in ['train', 'validation', 'test']]
    base = pipeline().fit(train['text'], train['label'])
    classifier = CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid').fit(val['text'], val['label'])
    subtype_rows = train[train['scam_type'].isin(TYPES)]
    subtype = pipeline().fit(subtype_rows['text'], subtype_rows['scam_type'])
    artifact = {'classifier': classifier, 'subtype': subtype, 'labels': LABELS, 'subtypes': TYPES, 'metadata': {'architecture': 'character + word TF-IDF / Logistic Regression', 'calibration': 'sigmoid on group-disjoint validation set', 'subtype_calibration': 'uncalibrated; synthetic and weak labels', 'training_seconds': round(time.perf_counter()-start, 2)}}
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    joblib.dump(artifact, ROOT / 'artifacts/lr.joblib', compress=3)
    probabilities = classifier.predict_proba(test['text'])
    probabilities = probabilities[:, [list(classifier.classes_).index(k) for k in LABELS]]
    result = report(test, probabilities, 'lr', artifact['metadata'])
    from sklearn.metrics import classification_report
    for method in ['synthetic_template_label', 'source_phishing_label']:
        selected = test[test['scam_type'].isin(TYPES) & (test['annotation_method'] == method)]
        if len(selected):
            result.setdefault('subtype_evaluation', {})[method] = {'n': len(selected), 'report': classification_report(selected['scam_type'], subtype.predict(selected['text']), labels=TYPES, output_dict=True, zero_division=0)}
    save_report(result, ROOT / 'reports/lr_metrics.json')

if __name__ == '__main__':
    main()
