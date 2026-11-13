import importlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
try:
    knowledge = importlib.import_module('inspection_knowledge')
except ModuleNotFoundError:
    knowledge = None


class RoutingTests(unittest.TestCase):
    def test_routes_fact_knowledge_and_mixed_requests(self):
        self.assertIsNotNone(knowledge, 'Missing inspection question router')
        cases = [
            ('当前区域变化像素前3名', 'data'),
            ('第二张呢？', 'data'),
            ('解释一下碎片化为什么需要复核', 'knowledge'),
            ('什么是连通域？', 'knowledge'),
            ('按操作手册，这张样本应该怎样检查？', 'mixed'),
            ('找出变化最大的3个待复核样本，解释复核依据', 'mixed'),
            ('当前样本为什么需要复核？', 'mixed'),
            ('生成当前区域报告，并附检查依据', 'mixed'),
            ('你好', 'general'),
            ('帮我做一份曲奇配方', 'general'),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(knowledge.route_question(question, has_context=True), expected)


class KnowledgeSearchTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(knowledge, 'Missing knowledge retriever')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.doc = self.folder / 'review.md'
        self.doc.write_text('---\ntitle: 质量检查\nsource: tools/quality.py\nversion: test-v1\n---\n'
                            '# 质量检查\n\n## 碎片化检查\n多个小连通域会形成碎片化。'
                            '结合前后时相影像检查噪声和真实小目标。\n\n'
                            '## 空洞检查\n空洞是预测区域内部未覆盖的背景孔洞，'
                            '应检查漏检与真实背景。\n', encoding='utf-8')

    def test_retrieves_relevant_section_with_source_and_version(self):
        hits = knowledge.KnowledgeIndex(self.folder).search('碎片化和小连通域应该检查什么？')
        self.assertTrue(hits)
        self.assertEqual(hits[0]['section'], '碎片化检查')
        self.assertEqual(hits[0]['source'], 'tools/quality.py')
        self.assertEqual(hits[0]['version'], 'test-v1')
        self.assertIn('真实小目标', hits[0]['text'])
        self.assertTrue(hits[0]['chunkId'])

    def test_unrelated_question_returns_no_evidence(self):
        self.assertEqual(knowledge.KnowledgeIndex(self.folder).search('量子纠缠与贝尔不等式'), [])

    def test_document_edit_invalidates_index(self):
        index = knowledge.KnowledgeIndex(self.folder)
        before = index.search('碎片化')[0]
        self.doc.write_text(self.doc.read_text(encoding='utf-8').replace('test-v1', 'test-v2')
                            .replace('结合前后时相影像', '新版本要求对照叠加图与前后时相影像'), encoding='utf-8')
        after = index.search('碎片化')[0]
        self.assertEqual(after['version'], 'test-v2')
        self.assertNotEqual(before['chunkId'], after['chunkId'])

    def test_citations_use_only_retrieved_sources(self):
        hits = knowledge.KnowledgeIndex(self.folder).search('碎片化')
        self.assertTrue(knowledge.valid_citations('建议核对小连通域。[1]', hits))
        self.assertFalse(knowledge.valid_citations('根据资料[9]，直接通过。', hits))
        self.assertFalse(knowledge.valid_citations('没有引用的回答', hits))


if __name__ == '__main__':
    unittest.main()
