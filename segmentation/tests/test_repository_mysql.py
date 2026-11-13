import copy
import importlib
import tempfile
import unittest
import uuid

from segmentation.tests.fixtures import manifest_fixture

try:
    repository = importlib.import_module('segmentation.repository')
except ModuleNotFoundError:
    repository = None


class RepositoryPresence(unittest.TestCase):
    def test_real_mysql_repository_exists(self):
        self.assertTrue(callable(getattr(repository, 'connect', None)), 'MySQL repository is not implemented')


@unittest.skipIf(repository is None, 'repository not implemented')
class MySQLTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = manifest_fixture(self.tmp.name)
        self.db = repository.connect('xunbian_backend_test')
        repository.migrate(self.db)
        self.batch_ids = [self.data['batch_id']]

    def tearDown(self):
        self.db.rollback()
        with self.db.cursor() as cur:
            for batch in self.batch_ids:
                for table in ['segmentation_assets', 'segmentation_predictions']:
                    cur.execute(f'DELETE a FROM {table} a JOIN segmentation_samples s ON s.sample_id=a.sample_id WHERE s.batch_id=%s', (batch,))
                cur.execute('DELETE FROM segmentation_evaluations WHERE batch_id=%s', (batch,))
                cur.execute('DELETE FROM segmentation_samples WHERE batch_id=%s', (batch,))
                cur.execute('DELETE FROM segmentation_batches WHERE batch_id=%s', (batch,))
            cur.execute('DELETE m FROM segmentation_models m LEFT JOIN segmentation_batches b ON b.model_id=m.model_id WHERE m.model_id=%s AND b.batch_id IS NULL', (self.data['model']['model_id'],))
        self.db.commit(); self.db.close(); self.tmp.cleanup()

    def batch_count(self, batch=None):
        with self.db.cursor() as cur:
            cur.execute('SELECT COUNT(*) AS n FROM segmentation_batches WHERE batch_id=%s', (batch or self.data['batch_id'],))
            return cur.fetchone()['n']

    def test_import_is_idempotent_with_regular_metric_columns(self):
        self.assertTrue(repository.ingest(self.db, self.data)['inserted'])
        self.assertFalse(repository.ingest(self.db, self.data)['inserted'])
        self.assertEqual(self.batch_count(), 1)
        with self.db.cursor() as cur:
            cur.execute('SELECT p.foreground_pixels,p.foreground_ratio FROM segmentation_predictions p JOIN segmentation_samples s ON s.sample_id=p.sample_id WHERE s.batch_id=%s', (self.data['batch_id'],))
            row = cur.fetchone()
        self.assertEqual(row['foreground_pixels'], 4)
        self.assertAlmostEqual(row['foreground_ratio'], .25)

    def test_existing_batch_cannot_be_rewritten(self):
        repository.ingest(self.db, self.data)
        changed = copy.deepcopy(self.data); changed['label'] = 'different'
        with self.assertRaises(ValueError):
            repository.ingest(self.db, changed)
        self.assertEqual(self.batch_count(), 1)

    def test_tampered_asset_is_rejected_before_insertion(self):
        from pathlib import Path
        Path(self.data['samples'][0]['assets'][0]['path']).write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            repository.ingest(self.db, self.data)
        self.assertEqual(self.batch_count(), 0)

    def test_database_error_rolls_back_whole_second_batch(self):
        repository.ingest(self.db, self.data)
        second = copy.deepcopy(self.data)
        second['batch_id'] = str(uuid.uuid4())
        second['samples'][0]['sample_id'] = str(uuid.uuid4())
        self.batch_ids.append(second['batch_id'])
        # The same asset primary keys collide after the batch/sample insert.
        with self.assertRaises(repository.pymysql.IntegrityError):
            repository.ingest(self.db, second)
        self.assertEqual(self.batch_count(second['batch_id']), 0)

    def test_foreign_key_prevents_orphan_prediction(self):
        with self.assertRaises(repository.pymysql.IntegrityError), self.db.cursor() as cur:
            cur.execute('INSERT INTO segmentation_predictions(sample_id,foreground_pixels,foreground_ratio,component_count,inference_ms) VALUES(%s,0,0,0,0)', (str(uuid.uuid4()),))
        self.db.rollback()


if __name__ == '__main__':
    unittest.main()
