import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
api = importlib.import_module('serve_change_agent_api')
try:
    static = importlib.import_module('static_facts')
except ModuleNotFoundError:
    static = None


class SqlFastPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / 'app_data.json'
        rows = [
            {'id': 'A', 'stem': 'A', 'changedPx': 1000, 'objectCount': 2, 'gridPosition': '左上', 'reviewStatus': 'low', 'reviewRequired': False},
            {'id': 'B', 'stem': 'B', 'changedPx': 900, 'objectCount': 4, 'gridPosition': '中部', 'reviewStatus': 'high', 'reviewRequired': True},
            {'id': 'C', 'stem': 'C', 'changedPx': 800, 'objectCount': 3, 'gridPosition': '右下', 'reviewStatus': 'medium', 'reviewRequired': True},
            {'id': 'D', 'stem': 'D', 'changedPx': 700, 'objectCount': 8, 'gridPosition': '中部', 'reviewStatus': 'low', 'reviewRequired': False},
        ]
        self.data.write_text(json.dumps({'regions': [{'id': 'test', 'short': '测试', 'items': rows}]}), encoding='utf-8')
        api.ARGS = SimpleNamespace(use_deepseek=False, use_qwen=False, static_sql=True,
                                   web_data=str(self.data), mysql_host=None, mysql_user=None,
                                   enable_rule_answer=True, enable_local_intent_corpus=False,
                                   model_id='test', prompt_version='test', max_new_tokens=256)
        self.addCleanup(api.TRANSIENT_SESSION_CONTEXTS.clear)
        self.addCleanup(api.TRANSIENT_QA_MESSAGES.clear)

    def payload(self, question):
        return {'question': question, 'region': {'id': 'test', 'short': '测试'},
                'sessionKey': self.id(), 'transientSession': True}

    def test_template_ranking_never_loads_llm(self):
        with patch.object(api, 'deepseek_text_answer', side_effect=AssertionError('Unnecessary LLM inference')):
            result = api.sql_agent_run(self.payload('当前区域变化像素前3名'))
        self.assertEqual(result['llmCalls'], 0)
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['A', 'B', 'C'])
        self.assertEqual(result['dataSource'], 'static_snapshot')
        self.assertIn('1,000', result['answer'])

    def test_selected_top_sample_does_not_turn_fresh_rank_into_second(self):
        payload = self.payload('当前区域变化像素前3名') | {'sample':{'id':'A'}}
        result = api.sql_agent_run(payload)
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['A','B','C'])

    def test_review_filter_applies_before_ranking(self):
        result = api.sql_agent_run(self.payload('当前区域变化像素前2名待复核样本'))
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['B', 'C'])

    def test_negative_review_filter_is_not_inverted(self):
        result = api.sql_agent_run(self.payload('当前区域无需复核的样本变化像素前2名'))
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['A', 'D'])

    def test_chinese_topk_is_respected(self):
        result = api.sql_agent_run(self.payload('当前区域变化像素前三名'))
        self.assertEqual(len(result['sqlRows']), 3)
        self.assertIn('3.', result['answer'])

    def test_high_risk_filter_does_not_include_medium(self):
        result = api.sql_agent_run(self.payload('当前区域高风险样本变化像素前3名'))
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['B'])

    def test_negative_review_list_does_not_add_positive_filter(self):
        result = api.sql_agent_run(self.payload('当前区域无需复核的样本有哪些？'))
        self.assertEqual({r['sample_id'] for r in result['sqlRows']}, {'A', 'D'})

    def test_numeric_filter_is_applied_before_sorting(self):
        result = api.sql_agent_run(self.payload('当前区域变化像素大于850的样本前3名'))
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['A', 'B'])

    def test_filtered_metric_does_not_replace_sort_metric(self):
        result = api.sql_agent_run(self.payload('当前区域对象数量大于3的样本变化像素前3名'))
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['B', 'D'])

    def test_non_high_risk_filter_does_not_mean_only_high(self):
        result = api.sql_agent_run(self.payload('当前区域非高风险样本变化像素前3名'))
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['A', 'C', 'D'])

    def test_private_tables_cannot_be_read_by_generated_sql(self):
        with patch.object(api, 'mysql_enabled', return_value=True), patch.object(api, 'mysql_connect', side_effect=AssertionError('Private query reached connection')):
            with self.assertRaises(ValueError):
                api.sql_agent_execute_sql('SELECT password_hash FROM auth_users')

    def test_comma_join_cannot_bypass_public_table_check(self):
        with patch.object(api, 'mysql_enabled', return_value=True), patch.object(api, 'mysql_connect', side_effect=AssertionError('Private join reached connection')):
            with self.assertRaises(ValueError):
                api.sql_agent_execute_sql('SELECT session_token FROM samples, auth_sessions')

    def test_followup_uses_previous_rank_instead_of_selected_sample(self):
        api.sql_agent_run(self.payload('当前区域变化像素前3名'))
        followup = self.payload('第二张呢？')
        followup['sample'] = {'id': 'D'}
        result = api.sql_agent_run(followup)
        self.assertEqual(result['sqlRows'][0]['sample_id'], 'B')
        self.assertEqual(result['answerScope'], 'sample')

    def test_static_source_rejects_writes(self):
        self.assertIsNotNone(static, 'Missing read-only static SQL adapter')
        store = static.StaticFacts(self.data)
        with self.assertRaises((ValueError, RuntimeError)):
            store.execute('DELETE FROM samples')
        with self.assertRaises((ValueError, RuntimeError)):
            store.execute("WITH x AS (SELECT 1) UPDATE samples SET changed_px=0")
        self.assertEqual(store.execute('SELECT COUNT(*) AS n FROM samples')['rows'][0]['n'], 4)


if __name__ == '__main__':
    unittest.main()
