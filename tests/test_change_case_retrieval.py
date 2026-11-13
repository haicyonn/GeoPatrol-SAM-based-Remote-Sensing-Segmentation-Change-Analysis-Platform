import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from change_case_retrieval import PairIndex, pair_features, RetrievalError


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(case_id=k, pair_hash=p, split='train') for k, p in [('b','bb'),('a','aa'),('c','cc')]]
        self.x = np.array([[1.,0.],[1.,0.],[0.,1.]], dtype=np.float32)

    def test_cosine_ties_and_duplicate_pair_exclusion(self):
        index = PairIndex(self.rows, self.x, {'encoder':'pinned'})
        self.assertEqual([x['case_id'] for x in index.search([1,0], 3)], ['a','b','c'])
        self.assertEqual([x['case_id'] for x in index.search([1,0], 3, exclude_pair='aa')], ['b','c'])
        self.assertAlmostEqual(index.search([10,0], 1)[0]['similarity'], 1.)

    def test_direction_preserved_and_invalid_vectors_rejected(self):
        a, b = np.array([[1.,0.]]), np.array([[0.,1.]])
        self.assertFalse(np.array_equal(pair_features(a,b), pair_features(b,a)))
        for bad in (np.zeros((3,2)), np.array([[np.nan,0],[1,0],[1,0]]), np.ones((2,2))):
            with self.assertRaises(RetrievalError): PairIndex(self.rows,bad,{})
        with self.assertRaises(RetrievalError): PairIndex(self.rows,self.x,{}).search([1,2,3],1)
        with self.assertRaises(RetrievalError): PairIndex(self.rows,self.x,{}).search([1,0],0)

    def test_cache_binds_source_ids_encoder_and_array_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'index'
            idx = PairIndex(self.rows,self.x,{'encoder':'pinned','source':'hash'})
            idx.save(path)
            loaded = PairIndex.load(path,self.rows,{'encoder':'pinned','source':'hash'})
            self.assertEqual(loaded.search([1,0],1)[0]['case_id'],'a')
            with self.assertRaises(RetrievalError): PairIndex.load(path,self.rows,{'encoder':'different'})
            with self.assertRaises(RetrievalError): PairIndex.load(path,self.rows[::-1],idx.identity)
            np.save(path/'vectors.npy', np.ones_like(self.x))
            with self.assertRaises(RetrievalError): PairIndex.load(path,self.rows,idx.identity)

    def test_only_training_case_rows_are_accepted(self):
        rows = [dict(self.rows[0],split='test')]
        with self.assertRaises(RetrievalError): PairIndex(rows,self.x[:1],{})

if __name__ == '__main__': unittest.main()
