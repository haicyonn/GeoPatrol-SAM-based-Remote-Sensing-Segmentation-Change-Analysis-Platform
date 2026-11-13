"""Local, source-preserving BM25 retrieval and inspection question routing.

This module uses no LLM, database, or network at import time. Knowledge text is
evidence, never executable instructions. BM25 is lexical retrieval, not a dense
embedding model; the response metadata deliberately names the actual method.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter
from pathlib import Path


DOMAIN = re.compile(r'变化|巡检|遥感|样本|复核|连通域|碎片|破碎|空洞|孔洞|漏检|误检|掩膜|配准|边界|像素|判读|对象|模型|报告|质量|面积|填充率|静态演示|实时数据库|实验指标|业务事实|deepseek|cdir|sd-osmr|mask|overlay', re.I)
EXPLAIN = re.compile(r'解释|为什么|为何|怎么|怎样|如何|依据|规范|手册|说明|什么是|是什么|区别|定义|原理|步骤|原因|注意|建议|检查什么')
SPECIFIC = re.compile(r'当前|这张|这幅|本图|该样本|这些|它们|它|第\s*[一二三四五六七八九十两\d]+|前\s*[一二三四五六七八九十两\d]+|top\s*\d+|找出|列出|筛选|排名|排序|最大|最多|最少|统计|多少|几个|比较|对比|相差|生成.*报告', re.I)
FOLLOWUP = re.compile(r'第二|第[一二三四五六七八九十\d]+|刚才|上一|那个|继续|它|它们')
ALIASES = {'孔洞': '空洞', '破碎': '碎片化', '碎片多': '碎片化', '连通区域': '连通域', '目标数': '对象数量', '配对对齐': '配准', '错检': '误检'}
STOP = set('什么 如何 怎么 为什么 为何 需要 应该 可以 进行 这个 那个 当前 样本 解释 一下 是否 哪些 多少 一些 根据 以及 以及 的是 用于 对于 一个 结果 建议 检查 问题 请问 我们 里面'.split())


def route_question(question: str, has_context: bool = False) -> str:
    q = str(question or '').strip()
    if not q:
        return 'general'
    domain = bool(DOMAIN.search(q))
    knowledge = bool(EXPLAIN.search(q))
    if re.search(r'按.*(规范|手册)|根据.*(规范|手册)', q):
        domain = True
    if domain and knowledge:
        return 'mixed' if SPECIFIC.search(q) else 'knowledge'
    if domain:
        return 'data'
    if has_context and FOLLOWUP.search(q):
        return 'mixed' if knowledge else 'data'
    return 'general'


def data_question(question: str) -> str:
    """Keep narrative metric names out of SQL intent and followup resolution."""
    q = str(question or '').strip()
    q = re.split(r'[，,；;]?\s*(?:并且?|同时|然后)?\s*(?:解释|说明|给出|提供|附上|附带|参考|结合)(?=.{1,80})', q, maxsplit=1)[0].strip(' ，,；;')
    if not q:
        q = str(question or '').strip()
    # A referential "second one, why review?" keeps the earlier list identity;
    # its explanation must not create a new review ranking.
    q = re.split(r'为什么|为何|应该怎样|应该如何|如何检查|怎样检查', q, maxsplit=1)[0].strip(' ，,；;') or q
    return q


def tokenize(text: str) -> list[str]:
    value = str(text or '').lower()
    for old, new in ALIASES.items():
        value = value.replace(old, new)
    for stop in sorted(STOP, key=len, reverse=True):
        value = value.replace(stop, ' ')
    tokens = re.findall(r'[a-z][a-z0-9_-]*|\d+(?:\.\d+)?', value)
    for run in re.findall(r'[\u4e00-\u9fff]+', value):
        if len(run) == 1:
            continue
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class KnowledgeIndex:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._signature = None
        self._chunks: list[dict] = []
        self._counts: list[Counter] = []
        self._df = Counter()
        self._average_length = 1.0
        self._lock = threading.RLock()

    def _refresh(self) -> None:
        paths = sorted(self.directory.glob('*.md')) if self.directory.is_dir() else []
        signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        if signature == self._signature:
            return
        chunks = []
        for path in paths:
            raw = path.read_text(encoding='utf-8-sig')
            metadata = {}
            if raw.startswith('---\n'):
                header, separator, body = raw[4:].partition('\n---\n')
                if separator:
                    raw = body
                    metadata = {k.strip(): v.strip() for line in header.splitlines()
                                if ':' in line for k, v in [line.split(':', 1)]}
            title = metadata.get('title', path.stem)
            section = title
            parts = []

            def append_chunk():
                text = '\n'.join(parts).strip()
                if not text:
                    return
                # Long sections are divided at paragraph boundaries to bound prompts.
                for offset in range(0, len(text), 1200):
                    excerpt = text[offset:offset + 1200]
                    digest = hashlib.sha256((path.name + section + metadata.get('version', '') + excerpt).encode()).hexdigest()[:16]
                    chunks.append({'chunkId': digest, 'title': title, 'section': section,
                                   'source': metadata.get('source', path.name),
                                   'document': path.name, 'version': metadata.get('version', 'unversioned'),
                                   'text': excerpt})

            for line in raw.splitlines():
                heading = re.match(r'^#{1,6}\s+(.+)', line)
                if heading:
                    append_chunk()
                    section = heading.group(1).strip()
                    parts = []
                else:
                    parts.append(line)
            append_chunk()
        counts = [Counter(tokenize((c['section'] + '\n') * 3 + c['text'])) for c in chunks]
        self._chunks, self._counts = chunks, counts
        self._df = Counter(token for count in counts for token in count)
        self._average_length = sum(sum(c.values()) for c in counts) / max(1, len(counts)) or 1.0
        self._signature = signature

    def search(self, question: str, limit: int = 3) -> list[dict]:
        tokens = set(tokenize(question))
        if not tokens:
            return []
        with self._lock:
            self._refresh()
            total = len(self._chunks)
            ranked = []
            for chunk, counts in zip(self._chunks, self._counts):
                length = sum(counts.values())
                score = 0.0
                for token in tokens:
                    freq = counts.get(token, 0)
                    if not freq:
                        continue
                    idf = math.log(1 + (total - self._df[token] + 0.5) / (self._df[token] + 0.5))
                    score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * length / self._average_length))
                if score > 0:
                    ranked.append({**chunk, 'score': round(score, 4)})
            ranked.sort(key=lambda c: (-c['score'], c['chunkId']))
            return ranked[:max(1, min(int(limit), 6))]

    def status(self) -> dict:
        with self._lock:
            self._refresh()
            return {'method': 'bm25-cjk-bigram', 'chunks': len(self._chunks),
                    'documents': len(set(c['document'] for c in self._chunks))}


def valid_citations(answer: str, hits: list[dict]) -> bool:
    refs = [int(x) for x in re.findall(r'\[(\d+)\]', str(answer or ''))]
    return bool(refs) and all(1 <= n <= len(hits) for n in refs)


def public_citations(hits: list[dict], explanation=None) -> list[dict]:
    if explanation and explanation.get('generationStatus') in {'no_evidence','citation_fallback','numeric_fallback','claim_fallback'}:
        return []
    return [{'id': i, 'chunkId': hit['chunkId'], 'title': hit['title'],
             'section': hit['section'], 'source': hit['source'],
             'version': hit['version'], 'excerpt': hit['text'], 'score': hit['score']}
            for i, hit in enumerate(hits, 1)]


def extractive_answer(hits: list[dict]) -> str:
    if not hits:
        return '本地知识库未找到足够依据，暂时不能给出有出处的解释。可补充问题涉及的指标或操作步骤。'
    return '\n\n'.join(f"[{i}] {hit['section']}：{hit['text']}" for i, hit in enumerate(hits, 1))
