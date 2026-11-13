import unittest
from pathlib import Path
from unittest.mock import patch

import test_sql_fast_path as sql_tests
api, ROOT = sql_tests.api, sql_tests.ROOT


class RagApiTests(unittest.TestCase):
    setUp = sql_tests.SqlFastPathTests.setUp
    payload = sql_tests.SqlFastPathTests.payload

    def test_knowledge_question_does_not_query_samples(self):
        with patch.object(api, 'sql_agent_run', side_effect=AssertionError('Knowledge must not query SQL')):
            result = api.handle_chat(self.payload('解释一下碎片化为什么需要复核'))
        self.assertEqual(result['answerScope'], 'knowledge')
        self.assertTrue(result['citations'])
        self.assertEqual(result['llmCalls'], 0)
        self.assertIn('连通域', result['answer'])

    def test_mixed_question_combines_filtered_facts_and_sources(self):
        result = api.handle_chat(self.payload('找出当前区域变化像素前2名待复核样本，并解释复核依据'))
        self.assertEqual(result['answerScope'], 'mixed')
        self.assertEqual([x['sample_id'] for x in result['evidenceSamples']], ['B', 'C'])
        self.assertTrue(result['citations'])
        self.assertIn('900', result['answer'])
        self.assertEqual([x['tool'] for x in result['toolTrace']][:2], ['query_samples', 'search_knowledge'])

    def test_explanation_metric_does_not_change_data_query(self):
        result = api.handle_chat(self.payload('当前区域变化像素前3名，并解释对象数量是什么'))
        self.assertEqual([r['sample_id'] for r in result['evidenceSamples']], ['A', 'B', 'C'])

    def test_ordinal_explanation_keeps_previous_rank(self):
        api.handle_chat(self.payload('当前区域变化像素前3名'))
        result = api.handle_chat(self.payload('第二张为什么需要复核？') | {'sample': {'id': 'D'}})
        self.assertEqual(result['evidenceSamples'][0]['sample_id'], 'B')

    def test_ordinal_past_previous_list_does_not_select_new_sample(self):
        api.handle_chat(self.payload('当前区域高风险样本变化像素前3名'))
        result = api.handle_chat(self.payload('第二张为什么需要复核？') | {'sample': {'id': 'D'}})
        self.assertEqual(result['evidenceSamples'], [])
        self.assertIn('第 2', result['answer'])

    def test_missing_knowledge_is_explicit_and_not_grounded(self):
        api.ARGS.knowledge_dir = str(Path(self.temp.name) / 'empty')
        result = api.handle_chat(self.payload('什么是连通域？'))
        self.assertEqual(result['citations'], [])
        self.assertFalse(result['grounded'])
        self.assertIn('未找到', result['answer'])

    def test_model_failure_returns_evidence_instead_of_fake_generation(self):
        api.ARGS.use_deepseek = True
        with patch.object(api, 'deepseek_text_answer', side_effect=RuntimeError('test unavailable')):
            result = api.handle_chat(self.payload('什么是连通域？'))
        self.assertEqual(result['generationStatus'], 'model_unavailable')
        self.assertEqual(result['llmCalls'], 1)
        self.assertTrue(result['citations'])

    def test_invented_citation_is_replaced_with_source_excerpts(self):
        api.ARGS.use_deepseek = True
        with patch.object(api, 'deepseek_text_answer', return_value='按外部规定直接通过。[99]'):
            result = api.handle_chat(self.payload('什么是连通域？'))
        self.assertEqual(result['generationStatus'], 'citation_fallback')
        self.assertNotIn('[99]', result['answer'])

    def test_valid_model_explanation_keeps_its_citation(self):
        api.ARGS.use_deepseek = True
        with patch.object(api,'deepseek_text_answer',return_value='连通域是相互连接的像素区域。[1]'):
            result=api.handle_chat(self.payload('什么是连通域？'))
        self.assertEqual(result['generationStatus'],'generated')
        self.assertEqual(result['llmCalls'],1)
        self.assertIn('[1]',result['answer'])

    def test_generated_numeric_claim_must_be_in_provided_evidence(self):
        api.ARGS.use_deepseek = True
        with patch.object(api, 'deepseek_text_answer', return_value='该样本有999999像素变化。[1]'):
            result = api.handle_chat(self.payload('当前样本为什么需要复核？') | {'sample': {'id': 'B'}})
        self.assertEqual(result['generationStatus'], 'numeric_fallback')
        self.assertNotIn('999999', result['answer'])

    def test_report_count_and_area_units_cannot_bypass_numeric_check(self):
        api.ARGS.use_deepseek = True
        with patch.object(api,'deepseek_text_answer',return_value='本区域共有999999张切片，平均面积999999平方公里。[1]'):
            result=api.handle_report_summary({'reportSnapshot':{'totalCount':4,'changedCount':4,'reviewCount':2}})
        self.assertEqual(result['generationStatus'],'numeric_fallback')
        self.assertNotIn('999999',result['answer'])

    def test_report_returns_sources_and_snapshot_identity(self):
        payload = self.payload('生成当前区域报告')
        payload['reportSnapshot'] = {'totalCount': 4, 'changedCount': 4, 'reviewCount': 2,
                                     'representatives': [{'stem': 'B', 'changedPx': 900}],
                                     'reasonBuckets': [{'label': '碎片化', 'count': 2}]}
        result = api.handle_report_summary(payload)
        self.assertTrue(result['citations'])
        self.assertEqual(result['answerScope'], 'region')
        self.assertTrue(result['reportFingerprint'])
        self.assertIn('4', result['answer'])

    def test_report_keeps_manual_correction_count(self):
        result = api.handle_report_summary({'reportSnapshot':{'totalCount':4,'changedCount':4,
            'reviewCount':3,'riskBuckets':[{'key':'high','count':2},{'key':'medium','count':1}],
            'correctionCount':2}})
        self.assertIn('高风险 2 张', result['answer'])
        self.assertIn('2 条人工修订', result['answer'])


if __name__ == '__main__':
    unittest.main()
