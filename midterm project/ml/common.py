"""Shared preprocessing and versioned labels; keep inference equal to training."""
from functools import lru_cache
import hashlib
import html
from pathlib import Path
import re
import unicodedata

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['normal', 'spam', 'scam']
TYPES = ['investment', 'shopping', 'relationship', 'government', 'job_loan', 'account']
TYPE_NAMES = {
    'investment': {'zh': '投資詐騙', 'en': 'Investment scam'},
    'shopping': {'zh': '網路購物與消費詐騙', 'en': 'Shopping & consumer scam'},
    'relationship': {'zh': '情感與社交信任詐騙', 'en': 'Relationship & social trust scam'},
    'government': {'zh': '政府與公家機關冒充', 'en': 'Government impersonation'},
    'job_loan': {'zh': '求職與申貸詐騙', 'en': 'Job & loan scam'},
    'account': {'zh': '科技與帳號盜用', 'en': 'Technology & account theft'},
}

@lru_cache(maxsize=1)
def converter():
    return OpenCC('s2twp')

def normalize(text: str) -> str:
    text = html.unescape(unicodedata.normalize('NFKC', text))
    text = re.sub(r'<[^>]{1,500}>', ' ', text)
    text = converter().convert(text)
    # OpenCC leaves 賬/帳 variants distinct; use Taiwan's 帳 consistently.
    text = text.replace('賬', '帳')
    text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def group_fingerprint(text: str) -> str:
    text = normalize(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '<url>', text)
    text = re.sub(r'\b[^\s@]+@[^\s@]+\b', '<email>', text)
    text = re.sub(r'\d+', '#', text)
    return hashlib.sha256(re.sub(r'\W+', '', text).encode()).hexdigest()

def language(text: str) -> str:
    return 'zh' if len(re.findall(r'[\u3400-\u9fff]', text)) >= 3 else 'en'
