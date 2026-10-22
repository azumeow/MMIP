import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.app import app, availability
from ml.common import normalize, group_fingerprint, ROOT, LABELS
from ml.explain import evidence
from ml.evaluate import report

client = TestClient(app)

def test_traditional_and_simplified_have_same_normalization():
    assert normalize('您的账号异常，请输入验证码') == normalize('您的帳號異常，請輸入驗證碼')

def test_unicode_and_html_normalization():
    assert normalize('<p>ＡＢＣ&nbsp;１２３</p>\u200b') == 'ABC 123'

def test_numeric_and_link_variants_share_group():
    assert group_fingerprint('Send 300 to https://a.example/123') == group_fingerprint('Send 900 to https://b.example/456')

def test_evidence_offsets_refer_to_original_unicode():
    text = '📨 提醒：立即提供验证码，不要告訴任何人。'
    spans = evidence(text)
    assert len(spans) >= 2
    assert all(text[s['start']:s['end']] == s['text'] for s in spans)
    assert all(a['end'] <= b['start'] for a,b in zip(spans,spans[1:]))

def test_urls_are_only_reported_as_cues():
    spans = evidence('Please visit https://example.test/check')
    assert spans[0]['rule'] == 'external_url'
    assert 'not opened' in spans[0]['reason']['en']

@pytest.mark.parametrize('payload', [
    {'text':'   '}, {'text':'ab'}, {'text':'a'*12001},
    {'text':'message text', 'models':['unknown']},
    {'text':'message text', 'models':['lr','lr']},
    {'text':'message text', 'models':[]},
])
def test_api_rejects_invalid_input(payload):
    response = client.post('/api/analyze',json=payload)
    assert response.status_code == 422
    assert response.headers['cache-control'] == 'no-store'

def test_health_reports_actual_artifact_state():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['models'] == availability()
    assert response.json()['input_retention'] == 'none'

def test_metrics_probability_column_order():
    frame = pd.DataFrame({'label':LABELS,'language':['en']*3,'source':['test']*3,'annotation_method':['gold']*3})
    good = report(frame,np.array([[.98,.01,.01],[.01,.98,.01],[.01,.01,.98]]),'test')
    assert good['accuracy'] == 1
    assert good['log_loss'] < .03

def test_no_fake_inference_when_artifacts_missing(monkeypatch, tmp_path):
    import backend.app as module
    monkeypatch.setattr(module, 'ARTIFACTS', tmp_path)
    monkeypatch.setattr(module, '_models', {})
    response = client.post('/api/analyze', json={'text':'Please review this message','models':['lr','bert']})
    assert response.status_code == 503
    assert len(response.json()['detail']['errors']) == 2

def test_real_lr_inference_when_trained():
    if not availability()['lr']:
        pytest.skip('LR has not been trained; do not claim model readiness')
    response = client.post('/api/analyze', json={'text':'老師您好，這是期中報告，下週三開會。','models':['lr']})
    assert response.status_code == 200
    result = response.json()['results'][0]
    assert set(result['probabilities']) == set(LABELS)
    assert abs(sum(result['probabilities'].values())-1) < 1e-6
    assert result['confidence'] == max(result['probabilities'].values())

def test_dataset_splits_have_no_group_overlap():
    if not (ROOT/'data/processed/train.jsonl').is_file():
        pytest.skip('Public data preparation has not completed')
    frames=[pd.read_json(ROOT/'data/processed'/f'{s}.jsonl',lines=True) for s in ['train','validation','test']]
    for i in range(3):
        for j in range(i+1,3):
            assert not set(frames[i]['group']) & set(frames[j]['group'])
            assert not set(frames[i]['fingerprint']) & set(frames[j]['fingerprint'])
