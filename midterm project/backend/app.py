from contextlib import asynccontextmanager
from pathlib import Path
import json
import logging
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from ml.common import ROOT, LABELS, TYPE_NAMES, language, normalize
from ml.explain import evidence

ARTIFACTS = ROOT / 'artifacts'
_models = {}
_lock = threading.Lock()
_inference_lock = threading.Lock()
MAX_CHARS = 12000

class AnalysisRequest(BaseModel):
    text: str = Field(min_length=3, max_length=MAX_CHARS)
    models: list[str] = Field(default_factory=lambda: ['lr', 'bert'], min_length=1, max_length=2)

    @field_validator('text')
    @classmethod
    def meaningful(cls, value):
        if len(normalize(value)) < 3:
            raise ValueError('Please enter at least 3 non-whitespace characters.')
        return value

    @field_validator('models')
    @classmethod
    def supported(cls, value):
        if len(set(value)) != len(value) or any(x not in ('lr', 'bert') for x in value):
            raise ValueError('Models must be unique values from lr, bert.')
        return value

def availability():
    return {
        'lr': (ARTIFACTS / 'lr.joblib').is_file(),
        'bert': (ARTIFACTS / 'bert' / 'training_metadata.json').is_file(),
    }

def load_model(name):
    with _lock:
        if name not in _models:
            if not availability()[name]:
                raise FileNotFoundError(f'{name} is not trained yet')
            if name == 'lr':
                import joblib
                _models[name] = joblib.load(ARTIFACTS / 'lr.joblib')
            else:
                from ml.bert_model import BertPredictor
                _models[name] = BertPredictor(ARTIFACTS / 'bert')
        return _models[name]

app = FastAPI(title='MailGuard — bilingual scam detection', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'], allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

@app.middleware('http')
async def private_responses(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response

@app.get('/api/health')
def health():
    return {'status': 'ok', 'models': availability(), 'labels': LABELS, 'scam_types': TYPE_NAMES, 'input_retention': 'none', 'max_chars': MAX_CHARS}

@app.get('/api/metrics')
def metrics():
    result = {}
    for name in ('lr', 'bert'):
        path = ROOT / 'reports' / f'{name}_metrics.json'
        if path.is_file():
            result[name] = json.loads(path.read_text())
    return result

@app.post('/api/analyze')
def analyze(payload: AnalysisRequest):
    text = normalize(payload.text)
    output, errors = [], []
    # One BERT batch at a time bounds CPU/RAM usage for a classroom demo.
    with _inference_lock:
        for name in payload.models:
            start = time.perf_counter()
            try:
                model = load_model(name)
                if name == 'lr':
                    probabilities = dict(zip(model['classifier'].classes_, map(float, model['classifier'].predict_proba([text])[0])))
                    subtype_prob = dict(zip(model['subtype'].classes_, map(float, model['subtype'].predict_proba([text])[0])))
                    truncated = False
                else:
                    probabilities, subtype_prob, truncated = model.predict(text)
                label = max(probabilities, key=probabilities.get)
                subtype = max(subtype_prob, key=subtype_prob.get) if label == 'scam' else None
                spans = evidence(payload.text) if label == 'scam' else []
                output.append({
                    'model': name, 'label': label, 'confidence': probabilities[label],
                    'probabilities': {key: probabilities.get(key, 0.0) for key in LABELS},
                    'scam_type': subtype,
                    'scam_type_confidence': subtype_prob[subtype] if subtype else None,
                    'scam_type_probabilities': subtype_prob if subtype else None,
                    'evidence': spans, 'evidence_method': 'transparent_rules_v1',
                    'low_confidence': probabilities[label] < 0.65,
                    'truncated': truncated,
                    'latency_ms': round((time.perf_counter()-start)*1000),
                })
            except FileNotFoundError:
                errors.append({'model': name, 'code': 'not_trained', 'message': f'{name} model artifacts are missing. Run its training command.'})
            except Exception:
                logging.getLogger('mailguard').error('Model inference failed (%s); user input is not logged', name)
                errors.append({'model': name, 'code': 'inference_failed', 'message': 'Model could not run. Check the server configuration.'})
    if not output:
        raise HTTPException(status_code=503, detail={'message': 'No requested trained model is available.', 'errors': errors})
    return {'results': output, 'errors': errors, 'language': language(text), 'normalized_to_traditional': True, 'disagreement': len({r['label'] for r in output}) > 1, 'input_retention': 'none'}

# A single origin and one Python process are sufficient for the built React app.
DIST = ROOT / 'frontend' / 'dist'
if DIST.is_dir():
    app.mount('/assets', StaticFiles(directory=DIST / 'assets'), name='assets')

    @app.get('/')
    def index():
        return FileResponse(DIST / 'index.html')
