"""Build 300/60/60 image-caption examples from the single COCO-CN dataset."""
import argparse
import collections
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from opencc import OpenCC

if __package__:
    from .common import (ROOT, PROJECT_ROOT, SEED, SOURCE_REPO, SOURCE_REVISION, ARCHIVE_NAME,
                         ARCHIVE_SHA256, ARCHIVE_URL, SPLITS, PROMPTS, DHASH_DISTANCE,
                         UNSUPPORTED, QA_EXCLUSIONS, sha256, write_json, text_key, image_info, validate_data, write_preview)
else:
    from common import (ROOT, PROJECT_ROOT, SEED, SOURCE_REPO, SOURCE_REVISION, ARCHIVE_NAME,
                        ARCHIVE_SHA256, ARCHIVE_URL, SPLITS, PROMPTS, DHASH_DISTANCE,
                        UNSUPPORTED, QA_EXCLUSIONS, sha256, write_json, text_key, image_info, validate_data, write_preview)

ARCHIVE_ROOT = 'coco-cn-version1805v1.1'
CAPTION_FILE = f'{ARCHIVE_ROOT}/imageid.human-written-caption.txt'
TAGS_FILE = f'{ARCHIVE_ROOT}/imageid.human-written-tags.txt'


def download(url, target, attempts=3):
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=(15, 60))
            response.raise_for_status()
            if not response.content:
                raise ValueError('Empty response')
            temp = target.with_suffix(target.suffix + '.tmp')
            temp.write_bytes(response.content)
            temp.replace(target)
            return target
        except (requests.RequestException, OSError, ValueError) as exc:
            last = exc
            time.sleep(.5 * (attempt + 1))
    raise RuntimeError(f'Download failed: {url}: {last}')


def read_annotations(path):
    """Read explicitly named tar members; never execute source code/extract the archive."""
    captions = collections.defaultdict(list)
    with tarfile.open(path, 'r:gz') as archive:
        text = archive.extractfile(CAPTION_FILE).read().decode('utf-8')
        for index, line in enumerate(text.splitlines()):
            annotation_id, sentence = line.split('\t', 1)
            image_id = annotation_id.split('#', 1)[0]
            captions[image_id].append({'annotation_id': annotation_id, 'line_index': index,
                                       'original_caption': sentence.strip()})
        tags = {}
        for line in archive.extractfile(TAGS_FILE).read().decode('utf-8').splitlines():
            parts = line.split()
            tags[parts[0]] = parts[1:]
        splits, split_hashes = {}, {}
        for split in SPLITS:
            raw = archive.extractfile(f'{ARCHIVE_ROOT}/coco-cn_{split}.txt').read()
            splits[split] = raw.decode('utf-8').splitlines()
            split_hashes[split] = hashlib.sha256(raw).hexdigest()
    sets = [set(v) for v in splits.values()]
    if any(len(v) != len(set(v)) for v in splits.values()) or any(a & b for i, a in enumerate(sets) for b in sets[i+1:]):
        raise ValueError('Official COCO-CN split lists are not disjoint')
    return captions, tags, splits, split_hashes


def caption_allowed(sentence):
    return (12 <= len(sentence) <= 80 and not UNSUPPORTED.search(sentence)
            and bool(re.search(r'[\u4e00-\u9fff]', sentence))
            and not any(mark in sentence for mark in ['http', '<image>', '\ufffd']))


def image_url(image_id):
    if not re.fullmatch(r'COCO_(train2014|val2014)_\d{12}', image_id):
        raise ValueError(f'Unexpected source image ID: {image_id}')
    partition = image_id.split('_')[1]
    # This is the source image service used by COCO-CN, not a second caption dataset.
    return f'http://images.cocodataset.org/{partition}/{image_id}.jpg'


def fetch_image(candidate, cache):
    try:
        image_id, _ = candidate
        path = download(image_url(image_id), cache / 'images' / f'{image_id}.jpg')
        info = image_info(path)
        if min(info['width'], info['height']) < 256:
            return None, 'low_resolution'
        return (path, info), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def metadata_path(path):
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build(cache, output, workers):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite dataset: {output}; use a different --output-dir')
    archive = download(ARCHIVE_URL, cache / ARCHIVE_NAME)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Annotation archive checksum mismatch')
    captions, tags, source_splits, split_hashes = read_annotations(archive)
    convert = OpenCC('s2t').convert
    used_images, used_pixels, used_answers, selected_hashes = set(), set(), set(), []
    stats, failures = {}, []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.caption_build_', dir=output.parent) as tmp:
        staging = Path(tmp) / 'data'
        staging.mkdir()
        for split, (size, stem) in SPLITS.items():
            candidates = []
            for image_id in source_splits[split]:
                if image_id in QA_EXCLUSIONS:
                    continue
                eligible = [c for c in captions.get(image_id, []) if caption_allowed(c['original_caption'])]
                if eligible:
                    # Shorter human descriptions make fewer unsupported detailed claims.
                    eligible.sort(key=lambda c: (len(c['original_caption']), c['annotation_id']))
                    candidates.append((image_id, eligible))
            # Stable pseudo-random ranking by image ID: excluding one candidate must
            # not reshuffle every other image or depend on thread completion order.
            candidates.sort(key=lambda item: hashlib.sha256(
                f'{SEED}:{split}:{item[0]}'.encode('utf-8')).hexdigest())
            examples = []
            rejected = collections.Counter()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for start in range(0, len(candidates), workers * 2):
                    batch = candidates[start:start + workers * 2]
                    # map preserves candidate order, so network completion order cannot change sampling.
                    downloads = executor.map(lambda item: fetch_image(item, cache), batch)
                    for (image_id, options), (result, failure) in zip(batch, downloads):
                        if result is None:
                            rejected['image_download_or_resolution'] += 1
                            failures.append({'split': split, 'image_id': image_id, 'reason': failure})
                            continue
                        path, info = result
                        if image_id in used_images or info['pixel_sha256'] in used_pixels:
                            rejected['duplicate_image'] += 1
                            continue
                        hashed = int(info['image_dhash'], 16)
                        if any((hashed ^ old).bit_count() <= DHASH_DISTANCE for old in selected_hashes):
                            rejected['perceptual_near_duplicate'] += 1
                            continue
                        option = next((c for c in options if text_key(convert(c['original_caption'])) not in used_answers), None)
                        if option is None:
                            rejected['duplicate_answer'] += 1
                            continue
                        answer = convert(option['original_caption'])
                        reference_answers = list(dict.fromkeys(convert(c['original_caption']) for c in options))
                        relative = f'images/{split}/{image_id}.jpg'
                        destination = staging / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, destination)
                        position = len(examples)
                        prompt = PROMPTS[position % len(PROMPTS)] if split == 'train' else PROMPTS[0]
                        examples.append({
                            'id': f'{split}_{position+1:03d}', 'image': relative, 'prompt': prompt,
                            'answer': answer, 'reference_answers': reference_answers,
                            **info,
                            'source': {'dataset': 'COCO-CN', 'repo': SOURCE_REPO, 'revision': SOURCE_REVISION,
                                       'split': split, 'image_id': image_id,
                                       'split_file': f'{ARCHIVE_ROOT}/coco-cn_{split}.txt',
                                       'caption_file': CAPTION_FILE, 'caption_type': 'human_written',
                                       **option, 'image_url': image_url(image_id),
                                       'original_tags': tags.get(image_id, []),
                                       'text_conversion': 'OpenCC s2t; character conversion only, no LLM generation/translation'},
                        })
                        used_images.add(image_id); used_pixels.add(info['pixel_sha256'])
                        used_answers.add(text_key(answer)); selected_hashes.append(hashed)
                        if len(examples) % 20 == 0:
                            print(f'{split}: {len(examples)}/{size}', flush=True)
                        if len(examples) == size:
                            break
                    if len(examples) == size:
                        break
            if len(examples) != size:
                raise ValueError(f'Insufficient valid {split} examples: {len(examples)}/{size}')
            purpose = {'train': 'SFT training only', 'val': 'validation/checkpoint selection only',
                       'test': 'held-out evaluation only; never train on these answers'}[split]
            write_json(staging / f'{stem}.json', {'purpose': purpose, 'split': split, 'count': size,
                                                'language': 'zh-Hant', 'examples': examples})
            lines = [{'id': row['id'], 'image': row['image'], 'width': row['width'], 'height': row['height'],
                      'conversations': [{'from': 'human', 'value': '<image>\n' + row['prompt']},
                                        {'from': 'gpt', 'value': row['answer']}]} for row in examples]
            (staging / f'{stem}.jsonl').write_text(
                ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in lines), encoding='utf-8')
            stats[split] = {'official_images': len(source_splits[split]), 'eligible_images': len(candidates),
                            'selected': size, 'rejected_after_download': dict(rejected)}
        # The training meta deliberately contains only the 300 training images.
        for split, name in [('train', 'internvl_train_meta.json'), ('val', 'internvl_val_meta.json')]:
            size, stem = SPLITS[split]
            write_json(staging / name, {f'coco_cn_{split}': {
                'root': metadata_path(output) + '/', 'annotation': metadata_path(output / f'{stem}.jsonl'),
                'data_augment': False, 'max_dynamic_patch': 6, 'repeat_time': 1, 'length': size}})
        write_preview(staging)
        artifact_hashes = {p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()}
        manifest = {
            'optimization': 'Describe visible objects/actions without inventing objects, intentions, or events',
            'model': 'OpenGVLab/InternVL3_5-14B', 'planned_training': 'SFT + QLoRA; training not executed',
            'seed': SEED, 'counts': {split: size for split, (size, _) in SPLITS.items()},
            'source_dataset_count': 1,
            'source': {'dataset': 'COCO-CN', 'repo': SOURCE_REPO, 'revision': SOURCE_REVISION,
                       'url': f'https://huggingface.co/datasets/{SOURCE_REPO}',
                       'original_project': 'https://github.com/li-xirong/coco-cn',
                       'archive_url': ARCHIVE_URL, 'archive_sha256': sha256(archive),
                       'caption_file': CAPTION_FILE, 'official_split_list_sha256': split_hashes},
            'selection': {'caption_length_characters': [12, 80], 'excluded_caption_pattern': UNSUPPORTED.pattern,
                          'caption_selection': 'shortest eligible original human-written caption; tie by annotation ID',
                          'image_selection': 'sort eligible images by SHA-256 of seed:official_split:image_id; take first valid unique images',
                          'text_conversion': 'opencc-python-reimplemented==0.1.7, s2t; character conversion without regional paraphrasing',
                          'visual_qa_excluded_images': QA_EXCLUSIONS,
                          'min_image_side': 256, 'near_duplicate_rule': f'64-bit dHash distance must exceed {DHASH_DISTANCE}',
                          'same_image_other_caption': 'never counted as another example',
                          'eval_prompt': PROMPTS[0], 'training_prompts': PROMPTS,
                          'training_prompt_counts': {p: 60 for p in PROMPTS}},
            'statistics': stats, 'download_failures': failures,
            'files_sha256': artifact_hashes,
            'limitations': ['Human captions can still contain omissions or annotation errors.',
                           'Keyword filtering and perceptual hashes do not prove perfect visual grounding.',
                           'Human-written tags are partial; unlisted objects must not automatically be called absent.',
                           'Local split separation cannot rule out these public images occurring in model pretraining.'],
            'rights': {'dataset_card_label': 'MIT',
                       'image_rights': 'COCO-CN uses MS-COCO photographs. Image rights remain with original owners; do not relabel every photo MIT.',
                       'terms_url': 'https://cocodataset.org/#termsofuse'},
        }
        write_json(staging / 'sources.json', manifest)
        validation = validate_data(staging)
        write_json(staging / 'validation_report.json', validation)
        staging.replace(output)
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    print('SUCCESS:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=Path('/tmp/hw3_vlm_sources'))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be in 1..16')
    build(args.cache_dir, args.output_dir, args.workers)


if __name__ == '__main__':
    main()
