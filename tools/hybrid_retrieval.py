"""Local BM25, BGE/ONNX dense retrieval, RRF and optional cross encoding.

Optional providers are imported and loaded on the first nonempty search. Models
must already be in ``cache_dir``: request handling never downloads weights or
uploads corpus text. A failed provider is disabled for this index's lifetime;
``status()`` and each hit identify the actual surviving retrieval stages.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

try:
    from .inspection_knowledge import KnowledgeIndex, tokenize
except ImportError:
    from inspection_knowledge import KnowledgeIndex, tokenize


EMBEDDING_MODEL = 'BAAI/bge-small-zh-v1.5'
RERANKER_MODEL = 'BAAI/bge-reranker-base'
RRF_K = 60
_CACHE_SCHEMA = 1


def _create_embedding_model(cache_dir: Path):
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=str(cache_dir),
                         providers=['CPUExecutionProvider'], threads=2,
                         local_files_only=True)


def _create_reranker_model(cache_dir: Path):
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=RERANKER_MODEL, cache_dir=str(cache_dir),
                            providers=['CPUExecutionProvider'], threads=2,
                            local_files_only=True)


def _normalise_vectors(values, expected_count: int, dimension: int | None = None):
    vectors = []
    for value in values:
        vector = [float(component) for component in value]
        if not vector or not all(math.isfinite(component) for component in vector):
            raise ValueError('embedding vectors must be nonempty and finite')
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError('embedding vector dimension mismatch')
        norm = math.sqrt(sum(component * component for component in vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('embedding vectors must have a finite, positive norm')
        vectors.append([component / norm for component in vector])
        if len(vectors) > expected_count:
            raise ValueError('embedding vector count mismatch')
    if len(vectors) != expected_count:
        raise ValueError('embedding vector count mismatch')
    return vectors


class HybridKnowledgeIndex(KnowledgeIndex):
    """Source-preserving retrieval with truthful configured/active metadata.

    Exact cosine search scans all corpus vectors, independently of lexical
    matches. Each stage contributes at most 24 candidates; cross encoding is
    limited to the resulting union (at most 48). Recreate the index to retry a
    failed model after repairing its local installation.
    """

    def __init__(self, directory: str | Path, *, dense: bool = False,
                 rerank: bool = False, cache_dir: str | Path | None = None):
        super().__init__(directory)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.directory.parent / '.model-cache'
        self._dense_configured = bool(dense)
        self._rerank_configured = bool(rerank)
        self._embedding_model = None
        self._reranker_model = None
        self._model_signature = (EMBEDDING_MODEL, RERANKER_MODEL)
        self._content_signature = None
        self._corpus_hash = ''
        self._vectors = None
        self._active = {'dense': False, 'rerank': False}
        self._errors: dict[str, str] = {}
        self._cache_state = 'unused'
        self._cache_error = None

    def _refresh(self) -> None:
        paths = sorted(self.directory.glob('*.md')) if self.directory.is_dir() else []
        files = tuple((path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths)
        model_signature = (EMBEDDING_MODEL, RERANKER_MODEL)
        signature = (model_signature, files)
        if signature == self._content_signature:
            return
        if model_signature[0] != self._model_signature[0]:
            self._embedding_model = None
            self._errors.pop('dense', None)
        if model_signature[1] != self._model_signature[1]:
            self._reranker_model = None
            self._errors.pop('rerank', None)
        self._model_signature = model_signature
        # The parent parser preserves document/section/source/version and IDs;
        # content hashing strengthens its mtime/size invalidation contract.
        self._signature = None
        super()._refresh()
        self._content_signature = signature
        payload = {'schema': _CACHE_SCHEMA, 'model': EMBEDDING_MODEL,
                   'inputFormat': 'section-newline-text-v1', 'chunks': self._chunks}
        self._corpus_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                                      separators=(',', ':')).encode('utf-8')).hexdigest()
        self._vectors = None
        self._active = {'dense': False, 'rerank': False}
        self._cache_state = 'unused'
        self._cache_error = None

    @staticmethod
    def _input(chunk: dict) -> str:
        return chunk['section'] + '\n' + chunk['text']

    def _bm25(self, question: str, candidate_limit: int) -> list[tuple[int, float]]:
        tokens = set(tokenize(question))
        ranked = []
        total = len(self._chunks)
        for position, counts in enumerate(self._counts):
            length = sum(counts.values())
            score = 0.0
            for token in tokens:
                frequency = counts.get(token, 0)
                if frequency:
                    idf = math.log(1 + (total - self._df[token] + 0.5) / (self._df[token] + 0.5))
                    score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / self._average_length))
            if score > 0:
                ranked.append((position, score))
        ranked.sort(key=lambda item: (-item[1], self._chunks[item[0]]['chunkId']))
        return ranked[:candidate_limit]

    def _cache_path(self) -> Path:
        return self.cache_dir / 'vectors' / (self._corpus_hash + '.json')

    def _read_vectors(self):
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            # JSON contains numbers only. Never deserialize pickle/object arrays.
            if path.stat().st_size > 256 * 1024 * 1024:
                raise ValueError('vector cache exceeds 256 MiB')
            cached = json.loads(path.read_text(encoding='utf-8'))
            if (cached['schema'] != _CACHE_SCHEMA or cached['model'] != EMBEDDING_MODEL or
                    cached['corpusHash'] != self._corpus_hash or
                    cached['chunkIds'] != [chunk['chunkId'] for chunk in self._chunks]):
                raise ValueError('vector cache identity mismatch')
            vectors = _normalise_vectors(cached['vectors'], len(self._chunks))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._cache_error = f'{type(exc).__name__}: {str(exc)[:240]}'
            return None
        self._cache_state = 'disk'
        return vectors

    def _write_vectors(self, vectors) -> None:
        temporary = None
        try:
            path = self._cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {'schema': _CACHE_SCHEMA, 'model': EMBEDDING_MODEL,
                       'corpusHash': self._corpus_hash,
                       'chunkIds': [chunk['chunkId'] for chunk in self._chunks], 'vectors': vectors}
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.tmp',
                                             dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except (OSError, ValueError, TypeError) as exc:
            # A read-only cache must not turn successful inference into BM25.
            self._cache_error = f'{type(exc).__name__}: {str(exc)[:240]}'
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _dense(self, question: str, candidate_limit: int) -> list[tuple[int, float]]:
        if not self._dense_configured or 'dense' in self._errors:
            return []
        try:
            if self._embedding_model is None:
                self._embedding_model = _create_embedding_model(self.cache_dir)
            if self._vectors is None:
                self._vectors = self._read_vectors()
                if self._vectors is None:
                    self._vectors = _normalise_vectors(
                        self._embedding_model.embed([self._input(chunk) for chunk in self._chunks], batch_size=32),
                        len(self._chunks))
                    self._cache_state = 'built'
                    self._write_vectors(self._vectors)
            query = _normalise_vectors(self._embedding_model.query_embed(question), 1,
                                       len(self._vectors[0]))[0]
            ranked = [(position, sum(a * b for a, b in zip(query, vector)))
                      for position, vector in enumerate(self._vectors)]
            ranked.sort(key=lambda item: (-item[1], self._chunks[item[0]]['chunkId']))
            self._active['dense'] = True
            return ranked[:candidate_limit]
        except Exception as exc:
            self._errors['dense'] = f'{type(exc).__name__}: {str(exc)[:240]}'
            self._active['dense'] = False
            return []

    def _rerank(self, question: str, hits: list[dict]) -> list[dict]:
        if not self._rerank_configured or 'rerank' in self._errors or not hits:
            return hits
        try:
            if self._reranker_model is None:
                self._reranker_model = _create_reranker_model(self.cache_dir)
            scores = [float(value) for value in self._reranker_model.rerank(
                question, [self._input(hit) for hit in hits], batch_size=16)]
            if len(scores) != len(hits) or not all(math.isfinite(score) for score in scores):
                raise ValueError('reranker must return one finite score per candidate')
            for hit, score in zip(hits, scores):
                hit['scores']['rerank'] = score
                hit['score'] = score
            hits.sort(key=lambda hit: (-hit['score'], -hit['scores']['rrf'], hit['chunkId']))
            self._active['rerank'] = True
        except Exception as exc:
            self._errors['rerank'] = f'{type(exc).__name__}: {str(exc)[:240]}'
            self._active['rerank'] = False
        return hits

    def _method(self) -> str:
        method = 'bm25+dense-rrf' if self._active['dense'] else 'bm25-cjk-bigram'
        return method + '+cross-encoder' if self._active['rerank'] else method

    def search(self, question: str, limit: int = 3) -> list[dict]:
        question = str(question or '').strip()
        limit = max(1, min(int(limit), 6))
        with self._lock:
            self._refresh()
            self._active = {'dense': False, 'rerank': False}
            if not question or not self._chunks:
                return []
            candidate_limit = min(len(self._chunks), max(12, limit * 4))
            lexical, dense = self._bm25(question, candidate_limit), self._dense(question, candidate_limit)
            candidates: dict[int, dict] = {}
            for stage, ranking in [('bm25', lexical), ('dense', dense)]:
                for rank, (position, score) in enumerate(ranking, 1):
                    if position not in candidates:
                        candidates[position] = {**self._chunks[position], 'score': 0.0,
                                                'scores': {'bm25': None, 'dense': None, 'rrf': 0.0, 'rerank': None},
                                                'ranks': {'bm25': None, 'dense': None}}
                    hit = candidates[position]
                    hit['scores'][stage] = score
                    hit['ranks'][stage] = rank
                    hit['scores']['rrf'] += 1.0 / (RRF_K + rank)
            hits = list(candidates.values())
            for hit in hits:
                hit['score'] = hit['scores']['rrf'] if self._active['dense'] else hit['scores']['bm25']
            hits.sort(key=lambda hit: (-hit['score'], hit['chunkId']))
            hits = self._rerank(question, hits)
            for hit in hits:
                hit['retrieval'] = {'method': self._method(), 'corpusHash': self._corpus_hash,
                                    'embeddingModel': EMBEDDING_MODEL if self._active['dense'] else None,
                                    'rerankerModel': RERANKER_MODEL if self._active['rerank'] else None,
                                    'degraded': [{'stage': stage, 'error': error} for stage, error in self._errors.items()]}
            return hits[:limit]

    def status(self) -> dict:
        with self._lock:
            self._refresh()
            return {'method': self._method(), 'chunks': len(self._chunks),
                    'documents': len({chunk['document'] for chunk in self._chunks}),
                    'configured': {'dense': self._dense_configured, 'rerank': self._rerank_configured,
                                   'embeddingModel': EMBEDDING_MODEL, 'rerankerModel': RERANKER_MODEL},
                    'active': dict(self._active),
                    'degraded': [{'stage': stage, 'error': error} for stage, error in self._errors.items()],
                    'corpusHash': self._corpus_hash,
                    'cache': {'vectors': self._cache_state, 'error': self._cache_error},
                    'retryPolicy': 'failed providers disabled until index recreation'}
