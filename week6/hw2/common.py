"""Shared dataset validation. Paths are relative to the dataset JSON, never cwd."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / 'vlm_20_dataset/vlm_20_questions.json'
MODELS = {
    'internvl': ('OpenGVLab/InternVL3_5-14B', 'a1e37197b393ce9eec9df700fef65c11f4a6ffbd'),
    'qwen': ('Qwen/Qwen3-VL-32B-Instruct', '0cfaf48183f594c314753d30a4c4974bc75f3ccb'),
}
SYSTEM_PROMPT = 'You are a helpful assistant.'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def load_dataset(path=DEFAULT_DATASET):
    from PIL import Image
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    qs = data['questions']
    if len(qs) != 20 or len({q['id'] for q in qs}) != 20:
        raise ValueError('Expected exactly 20 unique questions')
    for q in qs:
        image = (path.parent / q['image_path']).resolve()
        if not image.is_relative_to(path.parent):
            raise ValueError(f'Image outside dataset directory: {image}')
        if sha256(image) != q['image_sha256']:
            raise ValueError(f'Image hash mismatch: {q["id"]}')
        with Image.open(image) as im:
            im.verify()
        if not q.get('prompt') or q.get('ground_truth') is None:
            raise ValueError(f'Incomplete question: {q["id"]}')
    return data


if __name__ == '__main__':
    data = load_dataset()
    print(f'PASS: {len(data["questions"])} questions, readable images, SHA-256 and references verified.')
