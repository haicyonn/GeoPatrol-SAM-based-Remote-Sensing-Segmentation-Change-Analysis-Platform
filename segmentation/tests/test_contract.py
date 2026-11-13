import copy
import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


try:
    contract = importlib.import_module('segmentation.contract')
except ModuleNotFoundError:
    contract = None


class ContractTests(unittest.TestCase):
    def function(self, name):
        value = getattr(contract, name, None)
        self.assertTrue(callable(value), f'{name} is not implemented')
        return value

    def test_mask_statistics_count_pixels_and_components(self):
        fn = self.function('mask_statistics')
        mask = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 0]], dtype=np.uint8)
        got = fn(mask)
        self.assertEqual(got['foreground_pixels'], 2)
        self.assertAlmostEqual(got['foreground_ratio'], 2 / 9)
        self.assertEqual(got['component_count'], 1)
        self.assertEqual((got['width'], got['height']), (3, 3))

    def test_nonbinary_mask_rejected(self):
        fn = self.function('mask_statistics')
        with self.assertRaises(ValueError):
            fn(np.array([[0, 127]], dtype=np.uint8))

    def test_micro_metrics_not_average_of_per_image_scores(self):
        fn = self.function('segmentation_metrics')
        got = fn({'tp': 3, 'fp': 1, 'fn': 2, 'tn': 10})
        self.assertAlmostEqual(got['building_iou'], .5)
        self.assertAlmostEqual(got['f1'], 2 / 3)
        self.assertAlmostEqual(got['precision'], .75)

    def test_canonical_digest_is_order_independent(self):
        fn = self.function('digest_json')
        self.assertEqual(fn({'a': 1, 'b': 2}), fn({'b': 2, 'a': 1}))
        with self.assertRaises(ValueError):
            fn({'x': float('nan')})

    def test_artifact_size_hash_and_dimensions_checked(self):
        record = self.function('asset_record')
        validate = self.function('validate_asset')
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / 'a.png'
            Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(path)
            asset = record(path, 'prediction_mask')
            validate(asset, width=4, height=3)
            with self.assertRaises(ValueError):
                validate(asset, width=3, height=4)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                validate(asset, width=4, height=3)

    def test_windows_paths_translate_for_wsl(self):
        fn = self.function('local_path')
        if __import__('os').name == 'nt':
            self.assertEqual(str(fn('H:/data/a.png')).replace('\\', '/'), 'H:/data/a.png')
        else:
            self.assertEqual(str(fn('H:/data/a.png')), '/mnt/h/data/a.png')
        with self.assertRaises(ValueError):
            fn('https://example.com/image.png')

    def test_duplicate_sample_ids_and_bad_ratios_rejected(self):
        fn = self.function('validate_manifest')
        for manifest in ({'schema_version': 1, 'samples': []}, {'schema_version': 0}):
            with self.assertRaises(ValueError):
                fn(manifest)

    def test_duplicate_real_sample_identity_rejected(self):
        from segmentation.tests.fixtures import manifest_fixture
        fn = self.function('validate_manifest')
        with tempfile.TemporaryDirectory() as tmp:
            data = manifest_fixture(tmp)
            data['samples'].append(copy.deepcopy(data['samples'][0]))
            with self.assertRaises(ValueError):
                fn(data)

    def test_ratio_must_match_pixel_count(self):
        from segmentation.tests.fixtures import manifest_fixture
        fn = self.function('validate_manifest')
        with tempfile.TemporaryDirectory() as tmp:
            data = manifest_fixture(tmp)
            data['samples'][0]['foreground_ratio'] = .9
            with self.assertRaises(ValueError):
                fn(data)


if __name__ == '__main__':
    unittest.main()
