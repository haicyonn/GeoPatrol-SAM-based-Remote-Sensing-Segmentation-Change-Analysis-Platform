import copy
import unittest
import test_query_population as fixtures


class QueryRAGTests(unittest.TestCase):
    setUp = fixtures.PopulationTests.setUp
    service = fixtures.PopulationTests.service
    ask = fixtures.PopulationTests.ask
    count_outputs = fixtures.PopulationTests.count_outputs

    def enable_retrieval(self, service):
        self.retrieval_questions = []
        hit = dict(title='质量检查手册', section='空洞检查', source='quality.md', version='v1',
                   chunkId='quality#holes', text='内部空洞是检查线索，应核对前后时相和原始边界，不能直接认定漏检。', score=1.0)
        def retrieve(q):
            self.retrieval_questions.append(q)
            return [copy.deepcopy(hit)]
        service.retrieve = retrieve

    def test_knowledge_only_returns_real_sources_without_sql_generation(self):
        s = self.service([]); self.enable_retrieval(s)
        r = self.ask(s, '什么是空洞，应该怎么检查？')
        self.assertEqual(r['status'], 'knowledge_completed')
        self.assertEqual(r['sqlQuery'], '')
        self.assertEqual(r['citations'][0]['title'], '质量检查手册')
        self.assertTrue(r['knowledge']['answer'])

    def test_contextual_explanation_preserves_population_for_later_average(self):
        s = self.service(self.count_outputs()); self.enable_retrieval(s)
        self.ask(s, 'LEVIR中同时有空洞和碎片标记的样本共有多少张？')
        r = self.ask(s, '这些样本应该怎么检查？')
        self.assertEqual(r['status'], 'knowledge_completed')
        self.assertEqual(r['knowledge']['facts'][0]['sample_count'], 2)
        follow = self.service([fixtures.plan(columns=['mean']), 'SELECT AVG(changed_px) mean FROM samples'])
        self.assertEqual(self.ask(follow, '这些样本平均变化像素是多少')['sqlRows'], [{'mean': 20.0}])

    def test_mixed_query_attaches_evidence_to_actual_query_result(self):
        s = self.service(self.count_outputs()); self.enable_retrieval(s)
        r = self.ask(s, 'LEVIR中同时有空洞和碎片标记的样本共有多少张？并说明检查依据。')
        self.assertEqual(r['status'], 'completed')
        self.assertEqual(r['sqlRows'], [{'n': 2}])
        self.assertIn('knowledge', r)
        self.assertEqual(r['knowledge']['facts'][0]['sample_count'], 2)
        self.assertEqual(r['citations'][0]['section'], '空洞检查')

    def test_retrieval_failure_keeps_valid_sql_and_reports_unavailable(self):
        s = self.service(self.count_outputs())
        def broken(q): raise RuntimeError('retrieval unavailable')
        s.retrieve = broken
        r = self.ask(s, 'LEVIR中同时有空洞和碎片标记的样本共有多少张？并说明检查依据。')
        self.assertEqual(r['status'], 'completed')
        self.assertEqual(r['sqlRows'], [{'n': 2}])
        self.assertIn('knowledge', r)
        self.assertEqual(r['knowledge']['status'], 'unavailable')
        self.assertEqual(r['citations'], [])

    def test_data_change_during_retrieval_does_not_attach_stale_facts(self):
        s = self.service(self.count_outputs())
        def change(q):
            self.source.write_text(self.source.read_text(encoding='utf-8') + ' ', encoding='utf-8')
            return []
        s.retrieve = change
        r = self.ask(s, 'LEVIR中同时有空洞和碎片标记的样本共有多少张？并说明检查依据。')
        self.assertIn('knowledge', r)
        self.assertEqual(r['knowledge']['status'], 'stale')
        self.assertEqual(r['citations'], [])

    def test_maximum_with_explanation_still_executes_numeric_query(self):
        s=self.service(self.count_outputs()+[fixtures.plan(columns=['m']),'SELECT MAX(changed_px) m FROM samples'])
        self.enable_retrieval(s)
        self.ask(s,'LEVIR中同时有空洞和碎片标记的样本共有多少张？')
        r=self.ask(s,'给出这些样本的变化像素最大值并解释原因')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['sqlRows'],[{'m':30.0}])

    def start_list(self, s):
        return self.ask(s,'LEVIR按样本ID排序前2张，只给ID')

    def list_outputs(self):
        return [fixtures.plan(regions=['levir'],columns=['sample_id'],order=[dict(expression='sample_id',direction='ASC')],limit=2),
                "SELECT sample_id FROM samples WHERE region_id='levir' ORDER BY sample_id LIMIT 2"]

    def test_ordinal_knowledge_uses_single_sample_and_keeps_list_for_next_reference(self):
        s=self.service(self.list_outputs());self.enable_retrieval(s);self.start_list(s)
        r=self.ask(s,'刚才第2张为什么需要复核？')
        self.assertEqual(r['status'],'knowledge_completed');self.assertEqual(r['resolvedReference']['sampleId'],'B')
        self.assertEqual(r['knowledge']['facts'][0]['sample_count'],1)

    def test_mixed_ordinal_evidence_does_not_summarize_entire_list(self):
        s=self.service(self.list_outputs()+[fixtures.plan(columns=['changed_px']),"SELECT changed_px FROM samples WHERE sample_id='B'"])
        self.enable_retrieval(s);self.start_list(s)
        r=self.ask(s,'刚才第2张的变化像素是多少？并说明检查依据')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['sqlRows'],[{'changed_px':30.0}])
        self.assertEqual(r['knowledge']['facts'][0]['sample_count'],1)

    def test_generated_confirmed_error_diagnosis_falls_back_to_sources(self):
        s=self.service(self.count_outputs()+['这些样本的空洞已经确认是模型漏检建筑造成的。[1]'])
        self.enable_retrieval(s)
        self.ask(s,'LEVIR中同时有空洞和碎片标记的样本共有多少张？')
        r=self.ask(s,'这些样本应该怎么检查？')
        self.assertEqual(r['knowledge']['status'],'scope_fallback')
        self.assertNotIn('已经确认是模型漏检',r['knowledge']['answer'])

    def test_correct_population_count_can_be_used_in_generated_explanation(self):
        s=self.service(self.count_outputs()+['这些样本共有2张，应核对前后时相。[1]'])
        self.enable_retrieval(s)
        self.ask(s,'LEVIR中同时有空洞和碎片标记的样本共有多少张？')
        r=self.ask(s,'这些样本应该怎么检查？')
        self.assertEqual(r['knowledge']['status'],'generated')

    def test_unrelated_caution_does_not_authorize_confirmed_diagnosis(self):
        s=self.service(self.count_outputs()+['这些样本的空洞已经确认是模型漏检建筑造成的，检查时不能忽视原始影像。[1]'])
        self.enable_retrieval(s)
        self.ask(s,'LEVIR中同时有空洞和碎片标记的样本共有多少张？')
        r=self.ask(s,'这些样本应该怎么检查？')
        self.assertEqual(r['knowledge']['status'],'scope_fallback')


if __name__ == '__main__': unittest.main()
