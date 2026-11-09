"""Shared settings for the reproducible Qwen3-8B PPO experiment."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3-8B'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
REWARD_MODEL = 'Skywork/Skywork-Reward-V2-Qwen3-0.6B'
REWARD_REVISION = '8c14a4e9e6321deaf572544339b16b8d6bbe8886'
SYSTEM = 'You are a helpful assistant. Answer the user\'s question directly and concisely in English.'
SEED = 42


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all():
    import numpy as np
    import torch
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def actor_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def chat_ids(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)


def data_manifest():
    return {p.name:sha256(p) for p in sorted((ROOT/'data').glob('*.json')) if p.name!='sources.json'}
