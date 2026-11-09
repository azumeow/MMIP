"""Shared settings and integrity checks for the InternVL caption SFT dataset."""
import hashlib
import json
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DATA = ROOT / 'data'
MODEL = 'OpenGVLab/InternVL3_5-14B'
MODEL_REVISION = 'a1e37197b393ce9eec9df700fef65c11f4a6ffbd'
SEED = 42
SOURCE_REPO = 'AIMClab-RUC/COCO-CN'
SOURCE_REVISION = 'd40a7aac6fdfa88f5e30c0cd89516feaa8014739'
ARCHIVE_NAME = 'coco-cn-version1805v1.1.tar.gz'
ARCHIVE_SHA256 = '6c126cd8455363a404806e452ec75066a8fc96d73922d9357d993fcdd1d40b8a'
ARCHIVE_URL = f'https://huggingface.co/datasets/{SOURCE_REPO}/resolve/{SOURCE_REVISION}/{ARCHIVE_NAME}'
SPLITS = {'train': (300, 'train_sft_300'), 'val': (60, 'validation_60'), 'test': (60, 'target_test_60')}
PROMPTS = [
    '這張圖片在說什麼？',
    '請描述這張照片。',
    '照片中可以看到什麼？',
    '請說明圖片中的主要物體與動作。',
    '這張照片呈現了什麼場景？',
]
DHASH_DISTANCE = 4
# A reproducible heuristic, not a claim that all remaining human captions are flawless.
UNSUPPORTED = re.compile(
    r'似乎|好像|可能|也许|大概|估计|猜测|准备|将要|即将|打算|计划|等待|'
    r'思考|想要|希望|感到|觉得|快乐|开心|高兴|愉快|幸福|可爱|漂亮|美丽|'
    r'享受|著名|主人|夫妻|一家人|父亲|母亲|爸爸|妈妈|哥哥|姐姐|弟弟|妹妹|'
    r'生日|婚礼|庆祝|参加|比赛|参赛|聚会|聚餐|约会|度假|旅游|旅行|'
    r'情侣|恋人|朋友|同学|同事|家人|父子|母子|兄弟|姐妹|乘务员|服务员|'
    r'工作人员|运动员|歌手|新郎|新娘|慵懒|悠闲|忙碌|专心|认真|精美|精致|'
    r'古老|美味|好吃|享用|公家车|看起来|很久|很长时间|欧洲|亚洲|美洲|'
    r'他的|她的|自己的|自家|安静|宁静|吵闹|喧闹|交谈|聊天|讨论|教练')
QA_EXCLUSIONS = {
    'COCO_val2014_000000226408':
        'Visual spot-check: original human caption says lying on the ground, but the image shows a person on cushions/a sofa.',
    'COCO_val2014_000000080303':
        'Visual spot-check: original human caption calls the pictured combine harvester a tractor; exclude instead of rewriting the annotation.',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_key(value):
    return re.sub(r'[\s，。！？、；：,.!?;:]', '', value)


def dhash(image):
    from PIL import Image
    small = image.convert('L').resize((9, 8), Image.Resampling.LANCZOS)
    values = list(small.get_flattened_data() if hasattr(small, 'get_flattened_data') else small.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(values[y*9+x] > values[y*9+x+1])
    return f'{bits:016x}'


def image_info(path):
    from PIL import Image
    with Image.open(path) as source:
        source.verify()
    with Image.open(path) as source:
        image = source.convert('RGB')
        return {
            'width': image.width, 'height': image.height,
            'image_sha256': sha256(path),
            'pixel_sha256': hashlib.sha256(struct.pack('>II', *image.size) + image.tobytes()).hexdigest(),
            'image_dhash': dhash(image),
        }


def write_preview(data_dir):
    """Local HTML contact sheet; all image paths remain relative to data/."""
    import html
    data_dir = Path(data_dir)
    parts = ['<!doctype html><html lang="zh-Hant"><meta charset="utf-8">',
             '<title>COCO-CN 420 組 SFT 資料預覽</title>',
             '<style>body{font:16px sans-serif;margin:24px;background:#f4f4f4;color:#222}'
             '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}'
             'article{background:white;padding:14px;border-radius:8px}img{width:100%;height:240px;object-fit:contain}'
             '.original{color:#666;font-size:14px}code{font-size:12px;overflow-wrap:anywhere}</style>',
             '<h1>COCO-CN：train 300／val 60／test 60</h1>',
             '<p>單一公開資料集的原始圖片與人工 caption；下方可逐張核對。資料篩選不保證全部標註完全正確。</p>']
    for split, (size, stem) in SPLITS.items():
        parts.append(f'<h2>{split} — {size} 組</h2><div class="grid">')
        for row in read_json(data_dir / f'{stem}.json')['examples']:
            esc = html.escape
            parts.append(f'<article><b>{esc(row["id"])}</b><img loading="lazy" src="{esc(row["image"], quote=True)}">'
                         f'<p>{esc(row["prompt"])}</p><p><b>{esc(row["answer"])}</b></p>'
                         f'<p class="original">原文：{esc(row["source"]["original_caption"])}</p>'
                         f'<code>{esc(row["source"]["image_id"])}</code></article>')
        parts.append('</div>')
    parts.append('</html>')
    (data_dir / 'preview.html').write_text('\n'.join(parts), encoding='utf-8')


def validate_data(data_dir=DATA, verify_manifest=True):
    """Check real images, exact sizes, official-source metadata and JSON/JSONL parity."""
    data_dir = Path(data_dir).resolve()
    ids, source_ids, pixels, answers, hashes = set(), set(), set(), set(), []
    counts, images = {}, set()
    for split, (size, stem) in SPLITS.items():
        wrapper = read_json(data_dir / f'{stem}.json')
        rows = wrapper['examples']
        records = [json.loads(line) for line in (data_dir / f'{stem}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        if wrapper['count'] != size or len(rows) != size or len(records) != size:
            raise ValueError(f'Wrong {split} size: expected {size}')
        if wrapper['split'] != split:
            raise ValueError(f'Wrong split field: {split}')
        for row, record in zip(rows, records):
            source = row['source']
            if source['repo'] != SOURCE_REPO or source['revision'] != SOURCE_REVISION or source['split'] != split:
                raise ValueError(f'Wrong source/split on {row["id"]}')
            if source['caption_type'] != 'human_written':
                raise ValueError('Only original human-written captions are allowed')
            if row['id'] in ids or source['image_id'] in source_ids:
                raise ValueError('Repeated ID/image across splits')
            image_path = (data_dir / row['image']).resolve()
            if not image_path.is_relative_to(data_dir):
                raise ValueError('Image path escapes dataset directory')
            info = image_info(image_path)
            if any(row[k] != v for k, v in info.items()):
                raise ValueError(f'Image metadata/hash mismatch: {row["id"]}')
            if min(info['width'], info['height']) < 256:
                raise ValueError('Insufficient image resolution')
            if info['pixel_sha256'] in pixels or text_key(row['answer']) in answers:
                raise ValueError('Repeated image pixels or answer')
            value = int(info['image_dhash'], 16)
            if any((value ^ previous).bit_count() <= DHASH_DISTANCE for previous in hashes):
                raise ValueError('Perceptually near-duplicate selected images')
            if not row['answer'] or row['answer'] not in row['reference_answers']:
                raise ValueError('Missing answer/reference')
            if split != 'train' and row['prompt'] != PROMPTS[0]:
                raise ValueError('Evaluation prompt must be identical and fixed')
            expected = {
                'id': row['id'], 'image': row['image'], 'width': row['width'], 'height': row['height'],
                'conversations': [{'from': 'human', 'value': '<image>\n' + row['prompt']},
                                  {'from': 'gpt', 'value': row['answer']}],
            }
            if record != expected:
                raise ValueError(f'InternVL JSONL does not match readable JSON: {row["id"]}')
            ids.add(row['id']); source_ids.add(source['image_id']); pixels.add(info['pixel_sha256'])
            answers.add(text_key(row['answer'])); hashes.append(value); images.add(image_path)
        counts[split] = len(rows)
    actual = {p.resolve() for p in (data_dir / 'images').rglob('*') if p.is_file()}
    if actual != images:
        raise ValueError('Missing or unreferenced image files')
    if verify_manifest:
        manifest = read_json(data_dir / 'sources.json')
        if manifest['source']['repo'] != SOURCE_REPO or manifest['counts'] != counts:
            raise ValueError('Wrong source/counts in manifest')
        for filename, digest in manifest['files_sha256'].items():
            if sha256(data_dir / filename) != digest:
                raise ValueError(f'Stale dataset artifact: {filename}')
    return {'status': 'passed', 'counts': counts, 'total_images': len(images),
            'source_dataset_count': 1, 'unique_source_images': len(source_ids),
            'unique_decoded_images': len(pixels), 'unique_answers': len(answers),
            'minimum_pairwise_dhash_distance': min((a ^ b).bit_count() for i, a in enumerate(hashes) for b in hashes[i+1:]),
            'checks': ['counts', 'image_decode', 'file_sha256', 'pixel_sha256', 'dhash_near_duplicates',
                       'source_ids', 'answer_deduplication', 'json_jsonl_parity', 'single_source', 'fixed_eval_prompt']}


if __name__ == '__main__':
    print(json.dumps(validate_data(), ensure_ascii=False, indent=2))
