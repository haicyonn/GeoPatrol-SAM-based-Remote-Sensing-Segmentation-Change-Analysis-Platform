import importlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    predictor = importlib.import_module('segmentation.predict')
except ModuleNotFoundError:
    predictor = None


class SelectionTests(unittest.TestCase):
    def function(self):
        fn = getattr(predictor, 'select_images', None)
        self.assertTrue(callable(fn), 'Image selection is not implemented')
        return fn

    def test_manifest_order_and_limit_are_preserved(self):
        fn = self.function()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['a.png', 'b.png', 'c.png']:
                (root / name).touch()
            split = root / 'split.json'; split.write_text(json.dumps({'val': ['c.png', 'a.png', 'b.png']}))
            self.assertEqual([x.name for x in fn(root, split, 'val', 2)], ['c.png', 'a.png'])

    def test_missing_manifest_file_does_not_silently_shrink_batch(self):
        fn = self.function()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); split = root / 'split.json'
            split.write_text(json.dumps({'val': ['missing.png']}))
            with self.assertRaises(ValueError):
                fn(root, split, 'val', 1)

    def test_split_filename_cannot_escape_input_folder(self):
        fn = self.function()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); split = root / 'split.json'
            split.write_text(json.dumps({'val': ['../outside.png']}))
            with self.assertRaises(ValueError):
                fn(root, split, 'val', 1)

    def test_empty_image_folder_rejected(self):
        fn = self.function()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                fn(Path(tmp), None, 'val', 100)

    def test_release_positional_buffer_is_restored_from_adapter(self):
        fn = getattr(predictor, 'validate_base_state', None)
        self.assertTrue(callable(fn), 'Base checkpoint validation is not implemented')
        fn(['pe_layer.positional_encoding_gaussian_matrix', 'no_mask_embed.weight'],
           ['prompt_encoder.pe_layer.positional_encoding_gaussian_matrix'])

    def test_unexpected_missing_encoder_weight_is_rejected(self):
        fn = getattr(predictor, 'validate_base_state', None)
        self.assertTrue(callable(fn), 'Base checkpoint validation is not implemented')
        with self.assertRaises(ValueError):
            fn(['image_encoder.blocks.0.attn.qkv.weight'], [])

    def test_model_identity_covers_decoder_source_and_precision(self):
        fn = getattr(predictor, 'build_release_identity', None)
        self.assertTrue(callable(fn), 'Complete release identity is not implemented')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; (source / 'models').mkdir(parents=True)
            decoder = source / 'models/mask_decoder.py'; decoder.write_text('first')
            config = root / 'config.yaml'; config.write_text('fixed')
            a = fn(source, config, 'a' * 64, 'b' * 64, 'cuda', {'torch': 'test'})
            decoder.write_text('changed')
            b = fn(source, config, 'a' * 64, 'b' * 64, 'cuda', {'torch': 'test'})
            c = fn(source, config, 'a' * 64, 'b' * 64, 'cpu', {'torch': 'test'})
            self.assertNotEqual(a, b)
            self.assertNotEqual(b, c)
            self.assertIn('predict.py', a['inference_sha256'])

    def test_custom_labeled_images_do_not_claim_original_validation_score(self):
        fn = getattr(predictor, 'make_evaluation', None)
        self.assertTrue(callable(fn), 'Evaluation scope binding is not implemented')
        counts = dict(tp=3, fp=1, fn=2, tn=10)
        custom = fn(counts, 2, {'f1': .9}, False)
        self.assertEqual(custom['scope'], 'labeled_images')
        self.assertNotIn('saved_validation_f1', custom)
        matched = fn(counts, 100, {'f1': .9}, True)
        self.assertEqual(matched['scope'], 'validation_subset')
        self.assertIn('absolute_f1_delta_from_saved', matched)


if __name__ == '__main__':
    unittest.main()
