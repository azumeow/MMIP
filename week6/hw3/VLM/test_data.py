"""Dataset integrity and SFT export checks; no network, GPU or model needed."""
import json
import unittest

if __package__:
    from .common import DATA, SOURCE_REPO, SOURCE_REVISION, SPLITS, PROMPTS, read_json, validate_data, text_key
    from .prepare_data import caption_allowed, image_url
else:
    from common import DATA, SOURCE_REPO, SOURCE_REVISION, SPLITS, PROMPTS, read_json, validate_data, text_key
    from prepare_data import caption_allowed, image_url


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = {split: read_json(DATA / f'{stem}.json') for split, (_, stem) in SPLITS.items()}

    def test_actual_images_and_export_integrity(self):
        summary = validate_data()
        self.assertEqual(summary['counts'], {'train': 300, 'val': 60, 'test': 60})
        self.assertEqual(summary['unique_decoded_images'], 420)

    def test_one_dataset_and_original_splits(self):
        from opencc import OpenCC
        convert = OpenCC('s2t').convert
        for split, dataset in self.datasets.items():
            for row in dataset['examples']:
                self.assertEqual(row['source']['repo'], SOURCE_REPO)
                self.assertEqual(row['source']['revision'], SOURCE_REVISION)
                self.assertEqual(row['source']['split'], split)
                self.assertEqual(row['source']['caption_type'], 'human_written')
                self.assertIn('imageid.human-written-caption.txt', row['source']['caption_file'])
                self.assertTrue(row['source']['annotation_id'].startswith(row['source']['image_id'] + '#'))
                self.assertEqual(row['answer'], convert(row['source']['original_caption']))

    def test_no_image_or_answer_leakage(self):
        groups = [{r['source']['image_id'] for r in d['examples']} for d in self.datasets.values()]
        answers = [{text_key(r['answer']) for r in d['examples']} for d in self.datasets.values()]
        for i, first in enumerate(groups):
            for j in range(i+1, len(groups)):
                self.assertFalse(first & groups[j])
                self.assertFalse(answers[i] & answers[j])

    def test_prompt_balance_and_fixed_evaluation_prompt(self):
        rows = self.datasets['train']['examples']
        self.assertEqual({p: sum(r['prompt']==p for r in rows) for p in PROMPTS}, {p: 60 for p in PROMPTS})
        for split in ['val', 'test']:
            self.assertTrue(all(r['prompt']==PROMPTS[0] for r in self.datasets[split]['examples']))

    def test_training_meta_excludes_eval_samples(self):
        meta = read_json(DATA / 'internvl_train_meta.json')
        self.assertEqual(list(meta), ['coco_cn_train'])
        item = meta['coco_cn_train']
        self.assertEqual(item['length'], 300)
        self.assertTrue(item['annotation'].endswith('train_sft_300.jsonl'))
        self.assertNotIn('target_test', json.dumps(meta))
        self.assertNotIn('validation', json.dumps(meta))

    def test_caption_filter_and_source_image_url(self):
        self.assertTrue(caption_allowed('一只黄色的小狗趴在公园的长椅上。'))
        self.assertFalse(caption_allowed('一个男人准备去车站迎接他的父亲。'))
        self.assertFalse(caption_allowed('一只可爱的小狗开心地等待主人回来。'))
        self.assertFalse(caption_allowed('两位男子参加一场马术比赛。'))
        self.assertFalse(caption_allowed('公园的小路旁有一对情侣坐在长椅上。'))
        self.assertFalse(caption_allowed('猫。'))
        self.assertEqual(image_url('COCO_val2014_000000043734'),
                         'http://images.cocodataset.org/val2014/COCO_val2014_000000043734.jpg')
        with self.assertRaises(ValueError):
            image_url('../../wrong')


if __name__ == '__main__':
    unittest.main()
