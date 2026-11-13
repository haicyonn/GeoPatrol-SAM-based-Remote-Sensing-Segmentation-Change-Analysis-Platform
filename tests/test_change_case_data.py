import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from change_case_data import CaseDataError, prepare_corpus, public_record, safe_image


class CaseCorpusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'source'
        self.root.mkdir()
        self.rows = []

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, split, number, flag=1, colors=None):
        name = f'{split}_{number:06}.png'
        colors = colors or ((number, 10, 20), (number, 40, 70))
        for side, color in zip(('A', 'B'), colors):
            folder = self.root / 'images' / split / side
            folder.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (8, 8), color).save(folder / name)
        self.rows.append({'filename': name, 'split': split, 'filepath': split,
                          'changeflag': flag,
                          'sentences': [{'raw': 'SECRET reference ' + str(number)}]})
        return name

    def prepare(self, **kwargs):
        (self.root / 'LevirCCcaptions.json').write_text(json.dumps({'images': self.rows}))
        return prepare_corpus(self.root, Path(self.tmp.name) / 'prepared',
                              dev_size=2, test_size=2, **kwargs)

    def test_training_only_gallery_and_query_answers_are_separate(self):
        self.add('train', 1)
        for split, start in [('val', 10), ('test', 20)]:
            self.add(split, start, 0)
            self.add(split, start + 1, 1)
        meta = self.prepare()
        out = Path(self.tmp.name) / 'prepared'
        gallery = [json.loads(x) for x in (out / 'gallery.jsonl').read_text().splitlines()]
        queries = [json.loads(x) for x in (out / 'queries.jsonl').read_text().splitlines()]
        self.assertEqual([r['split'] for r in gallery], ['train'])
        self.assertEqual(len(queries), 4)
        for row in queries:
            self.assertNotIn('captions', row)
            self.assertNotIn('changeflag', row)
            self.assertNotIn('SECRET', json.dumps(row))
        self.assertEqual(meta['split_counts'], {'train': 1, 'val': 2, 'test': 2})
        gold = [json.loads(x) for x in (out / 'scoring_only.jsonl').read_text().splitlines()]
        self.assertEqual(len(gold), 4)
        self.assertTrue(all('captions' in row for row in gold))

    def test_reverse_pair_duplicate_never_becomes_a_holdout(self):
        a, b = ((1, 2, 3), (9, 8, 7))
        self.add('train', 1, colors=(a, b))
        dup = self.add('test', 20, 0, colors=(b, a))
        for split, start in [('val', 10), ('test', 30)]:
            self.add(split, start, 0)
            self.add(split, start + 1, 1)
        meta = self.prepare()
        self.assertEqual(meta['cross_split_exclusions'], 1)
        query_text = (Path(self.tmp.name) / 'prepared' / 'queries.jsonl').read_text()
        self.assertNotIn(dup, query_text)

    def test_unsafe_filename_and_symlink_escape_rejected(self):
        with self.assertRaises(CaseDataError):
            safe_image(self.root, 'train', 'A', '../../secret.png')
        with self.assertRaises(CaseDataError):
            safe_image(self.root, 'other', 'A', 'image.png')
        outside = Path(self.tmp.name) / 'outside.png'
        Image.new('RGB', (8, 8)).save(outside)
        folder = self.root / 'images' / 'train' / 'A'
        folder.mkdir(parents=True)
        try:
            (folder / 'linked.png').symlink_to(outside)
        except OSError:
            return
        with self.assertRaises(CaseDataError):
            safe_image(self.root, 'train', 'A', 'linked.png')

    def test_preparation_is_frozen_not_silently_overwritten(self):
        self.add('train', 1)
        for split, start in [('val', 10), ('test', 20)]:
            self.add(split, start, 0)
            self.add(split, start + 1, 1)
        self.prepare()
        with self.assertRaises(CaseDataError):
            self.prepare()

    def test_public_query_record_cannot_leak_reference_or_flags(self):
        record = {'case_id': 'test:x', 'split': 'test', 'filename': 'x.png',
                  'before_path': '/secret/A.png', 'after_path': '/secret/B.png',
                  'pair_hash': 'x', 'captions': ['SECRET'], 'changeflag': 1}
        value = public_record(record)
        self.assertNotIn('SECRET', json.dumps(value))
        self.assertNotIn('changeflag', value)
        self.assertNotIn('/secret', json.dumps(value))


if __name__ == '__main__':
    unittest.main()
