"""Retrieval algorithms run for real; only local ONNX providers are replaced."""
import importlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
try:
    hybrid = importlib.import_module('hybrid_retrieval')
except ModuleNotFoundError:
    hybrid = None


class Embeddings:
    def __init__(self, vectors=None, failure=None):
        self.vectors = vectors or {}
        self.failure = failure
        self.documents = []

    def embed(self, documents, **kwargs):
        documents = list(documents)
        self.documents.extend(documents)
        if self.failure:
            raise self.failure
        return iter(self.vectors[text.split('\n', 1)[0]] for text in documents)

    def query_embed(self, question, **kwargs):
        return iter([[1.0, 0.0]])


class CrossEncoder:
    def __init__(self, scores):
        self.scores = scores

    def rerank(self, question, documents, **kwargs):
        return iter(self.scores[text.split('\n', 1)[0]] for text in documents)


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(hybrid, 'Missing genuine hybrid retriever')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'knowledge'
        self.folder.mkdir()
        self.cache = Path(self.temp.name) / 'cache'

    def write(self, name, text, version='v1'):
        path = self.folder / f'{name}.md'
        path.write_text(f'---\ntitle: {name}\nsource: local/{name}.pdf#page=7\n'
                        f'version: {version}\n---\n# {name}\n{text}\n', encoding='utf-8')
        return path

    def index(self, **kwargs):
        return hybrid.HybridKnowledgeIndex(self.folder, cache_dir=self.cache, **kwargs)

    def test_lexical_default_preserves_sources_without_loading_models(self):
        self.write('碎片化', '多个连通域需要复核。')
        with patch.object(hybrid, '_create_embedding_model', side_effect=AssertionError('unexpected model load')):
            index = self.index()
            hits = index.search('碎片化')
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['source'], 'local/碎片化.pdf#page=7')
        self.assertEqual(hits[0]['version'], 'v1')
        self.assertEqual(hits[0]['document'], '碎片化.md')
        self.assertGreater(hits[0]['scores']['bm25'], 0)
        self.assertIsNone(hits[0]['scores']['dense'])
        self.assertEqual(index.status()['method'], 'bm25-cjk-bigram')
        self.assertEqual(index.search('量子纠缠'), [])

    def test_dense_retrieves_semantic_only_document(self):
        self.write('lexical', 'needle exact term')
        self.write('semantic', '建筑物边缘错位需要核对两个时相。')
        provider = Embeddings({'lexical': [0, 1], 'semantic': [1, 0]})
        with patch.object(hybrid, '_create_embedding_model', return_value=provider):
            index = self.index(dense=True)
            hits = index.search('needle', limit=2)
        semantic = next(hit for hit in hits if hit['section'] == 'semantic')
        self.assertIsNone(semantic['ranks']['bm25'])
        self.assertEqual(semantic['ranks']['dense'], 1)
        self.assertAlmostEqual(semantic['scores']['dense'], 1.0)
        self.assertTrue(index.status()['active']['dense'])

    def test_rrf_combines_independent_rankings_instead_of_adding_raw_scores(self):
        self.write('A', 'needle ' + 'filler ' * 20)
        self.write('B', 'needle ' * 5)
        self.write('C', 'semantic evidence')
        provider = Embeddings({'A': [1, 0], 'B': [0, 1], 'C': [.6, .8]})
        with patch.object(hybrid, '_create_embedding_model', return_value=provider):
            hits = self.index(dense=True).search('needle', limit=3)
        self.assertEqual([hit['section'] for hit in hits], ['A', 'B', 'C'])
        self.assertEqual(hits[0]['ranks'], {'bm25': 2, 'dense': 1})
        self.assertAlmostEqual(hits[0]['scores']['rrf'], 0.032522474881, places=10)

    def test_cross_encoder_reorders_fused_candidates_and_preserves_stage_scores(self):
        self.write('A', 'needle ' * 5)
        self.write('B', 'needle other')
        provider = Embeddings({'A': [1, 0], 'B': [.6, .8]})
        with patch.object(hybrid, '_create_embedding_model', return_value=provider), \
                patch.object(hybrid, '_create_reranker_model', return_value=CrossEncoder({'A': -2.0, 'B': 3.0})):
            index = self.index(dense=True, rerank=True)
            hits = index.search('needle', limit=1)
        self.assertEqual(hits[0]['section'], 'B')
        self.assertEqual(hits[0]['scores']['rerank'], 3.0)
        self.assertGreater(hits[0]['scores']['rrf'], 0)
        self.assertTrue(index.status()['active']['rerank'])

    def test_missing_embedding_falls_back_visibly_and_does_not_retry_each_request(self):
        self.write('lexical', 'needle evidence')
        with patch.object(hybrid, '_create_embedding_model', side_effect=RuntimeError('model unavailable')) as factory:
            index = self.index(dense=True)
            self.assertEqual(index.search('needle')[0]['section'], 'lexical')
            self.assertEqual(index.search('needle')[0]['section'], 'lexical')
            self.assertEqual(factory.call_count, 1, 'failed model initialization must be bounded')
        status = index.status()
        self.assertTrue(status['configured']['dense'])
        self.assertFalse(status['active']['dense'])
        self.assertEqual(status['method'], 'bm25-cjk-bigram')
        self.assertEqual(status['degraded'][0]['stage'], 'dense')
        self.assertIn('model unavailable', status['degraded'][0]['error'])

    def test_invalid_embedding_vectors_degrade_without_publishing_bad_cache(self):
        self.write('A', 'needle evidence')
        with patch.object(hybrid, '_create_embedding_model', return_value=Embeddings({'A': [math.nan, 1]})):
            index = self.index(dense=True)
            hits = index.search('needle')
        self.assertEqual(hits[0]['section'], 'A')
        self.assertFalse(index.status()['active']['dense'])
        self.assertIn('finite', index.status()['degraded'][0]['error'])
        self.assertFalse(list(self.cache.glob('vectors/*.json')))

    def test_rerank_failure_retains_real_fusion_and_names_actual_active_backend(self):
        self.write('A', 'needle evidence')
        provider = Embeddings({'A': [1, 0]})
        with patch.object(hybrid, '_create_embedding_model', return_value=provider), \
                patch.object(hybrid, '_create_reranker_model', side_effect=RuntimeError('reranker unavailable')):
            index = self.index(dense=True, rerank=True)
            hits = index.search('needle')
        self.assertEqual(hits[0]['section'], 'A')
        status = index.status()
        self.assertTrue(status['active']['dense'])
        self.assertFalse(status['active']['rerank'])
        self.assertEqual(status['method'], 'bm25+dense-rrf')
        self.assertEqual(status['degraded'][0]['stage'], 'rerank')

    def test_content_edit_invalidates_vectors_even_with_same_size_and_mtime(self):
        path = self.write('A', 'needle old evidence')
        provider = Embeddings({'A': [1, 0]})
        with patch.object(hybrid, '_create_embedding_model', return_value=provider):
            index = self.index(dense=True)
            before = index.search('needle')[0]
            stat = path.stat()
            path.write_text(path.read_text(encoding='utf-8').replace('old', 'new'), encoding='utf-8')
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            after = index.search('needle')[0]
        self.assertIn('new evidence', after['text'])
        self.assertNotEqual(before['chunkId'], after['chunkId'])
        self.assertEqual(len(provider.documents), 2)
        self.assertNotEqual(before['retrieval']['corpusHash'], after['retrieval']['corpusHash'])

    def test_valid_disk_vectors_reuse_corpus_embeddings_across_instances(self):
        self.write('A', 'needle evidence')
        first = Embeddings({'A': [1, 0]})
        second = Embeddings(failure=AssertionError('must reuse corpus vectors'))
        with patch.object(hybrid, '_create_embedding_model', return_value=first):
            self.index(dense=True).search('needle')
        with patch.object(hybrid, '_create_embedding_model', return_value=second):
            reopened = self.index(dense=True)
            hits = reopened.search('needle')
        self.assertAlmostEqual(hits[0]['scores']['dense'], 1.0)
        self.assertTrue(reopened.status()['active']['dense'])
        self.assertEqual(reopened.status()['cache']['vectors'], 'disk')

    def test_corrupt_cache_is_rebuilt_as_data_not_loaded_as_pickle(self):
        self.write('A', 'needle evidence')
        with patch.object(hybrid, '_create_embedding_model', return_value=Embeddings({'A': [1, 0]})):
            self.index(dense=True).search('needle')
        cached = next(self.cache.glob('vectors/*.json'))
        cached.write_text('{malformed cache', encoding='utf-8')
        with patch.object(hybrid, '_create_embedding_model', return_value=Embeddings({'A': [0, 1]})):
            index = self.index(dense=True)
            hits = index.search('needle')
        self.assertAlmostEqual(hits[0]['scores']['dense'], 0.0)
        self.assertTrue(index.status()['active']['dense'])
        self.assertIn('vectors', json.loads(cached.read_text(encoding='utf-8')))
        self.assertFalse(list(self.cache.glob('vectors/*.tmp')))

    def test_empty_question_and_corpus_do_not_load_optional_models(self):
        with patch.object(hybrid, '_create_embedding_model', side_effect=AssertionError('unexpected load')):
            index = self.index(dense=True, rerank=True)
            self.assertEqual(index.search(''), [])
            self.assertEqual(index.search('needle'), [])
        self.assertFalse(index.status()['active']['dense'])
        self.assertEqual(index.status()['chunks'], 0)

    def test_model_change_invalidates_existing_memory_and_disk_vectors(self):
        self.write('A', 'needle evidence')
        first = Embeddings({'A': [1, 0]})
        with patch.object(hybrid, '_create_embedding_model', return_value=first):
            index = self.index(dense=True)
            before = index.search('needle')[0]
        with patch.object(hybrid, 'EMBEDDING_MODEL', 'local/new-model-revision'), \
                patch.object(hybrid, '_create_embedding_model', return_value=Embeddings({'A': [0, 1]})):
            after = index.search('needle')[0]
        self.assertAlmostEqual(after['scores']['dense'], 0.0)
        self.assertNotEqual(before['retrieval']['corpusHash'], after['retrieval']['corpusHash'])

    def test_query_dimension_mismatch_falls_back_and_is_not_retried(self):
        self.write('A', 'needle evidence')
        with patch.object(hybrid, '_create_embedding_model', return_value=Embeddings({'A': [1, 0, 0]})):
            index = self.index(dense=True)
            hit = index.search('needle')[0]
            again = index.search('needle')[0]
        self.assertIsNone(hit['scores']['dense'])
        self.assertIsNone(again['scores']['dense'])
        self.assertIn('dimension', index.status()['degraded'][0]['error'])

    def test_bad_rerank_output_does_not_partly_replace_fusion_scores(self):
        self.write('A', 'needle evidence')
        self.write('B', 'needle ' * 5)
        with patch.object(hybrid, '_create_embedding_model', return_value=Embeddings({'A': [1, 0], 'B': [0, 1]})), \
                patch.object(hybrid, '_create_reranker_model', return_value=CrossEncoder({'A': 5, 'B': math.nan})):
            index = self.index(dense=True, rerank=True)
            hits = index.search('needle', limit=2)
        self.assertTrue(all(hit['score'] == hit['scores']['rrf'] for hit in hits))
        self.assertTrue(all(hit['scores']['rerank'] is None for hit in hits))
        self.assertFalse(index.status()['active']['rerank'])
        self.assertIn('finite', index.status()['degraded'][0]['error'])

    def test_cache_publish_failure_does_not_disable_successful_dense_search(self):
        self.write('A', 'needle evidence')
        with patch.object(hybrid, '_create_embedding_model', return_value=Embeddings({'A': [1, 0]})), \
                patch.object(hybrid.os, 'replace', side_effect=PermissionError('cache read only')):
            index = self.index(dense=True)
            hits = index.search('needle')
        self.assertAlmostEqual(hits[0]['scores']['dense'], 1.0)
        self.assertTrue(index.status()['active']['dense'])
        self.assertIn('cache read only', index.status()['cache']['error'])
        self.assertFalse(list(self.cache.glob('vectors/*.tmp')))

    def test_document_removal_never_reuses_deleted_evidence(self):
        path = self.write('A', 'needle evidence')
        self.write('B', 'other evidence')
        with patch.object(hybrid, '_create_embedding_model', return_value=Embeddings({'A': [1, 0], 'B': [0, 1]})):
            index = self.index(dense=True)
            self.assertEqual(len(index.search('needle', limit=3)), 2)
            path.unlink()
            hits = index.search('needle', limit=3)
        self.assertEqual([hit['document'] for hit in hits], ['B.md'])
        self.assertEqual(index.status()['documents'], 1)


if __name__ == '__main__':
    unittest.main()
