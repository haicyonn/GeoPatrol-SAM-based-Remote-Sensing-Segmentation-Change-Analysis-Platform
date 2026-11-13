import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
try:
    import inspection_workflow as workflow
except ImportError:
    workflow = None


class TaskPlanningTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(workflow, 'Stateful inspection workflow has not been implemented')
        self.regions = [{'id': 'test', 'short': '测试区域'}, {'id': 'whu', 'short': '武汉'}]
        self.payload = {'region': {'id': 'test'}}

    def parse(self, q):
        return workflow.parse_task(q, self.payload, self.regions)

    def test_colloquial_priority_request_is_a_query(self):
        plan = self.parse('找出风险最高的三张图')
        self.assertEqual(plan['mode'], 'data')
        self.assertEqual(plan['metric'], 'review_priority')
        self.assertEqual(plan['limit'], 3)

    def test_ambiguous_priority_requests_a_metric(self):
        plan = self.parse('哪些要优先看？')
        self.assertEqual(plan['issues'][0]['slot'], 'metric')

    def test_date_filter_is_not_silently_dropped(self):
        plan = self.parse('最近一个月高风险样本前3名')
        self.assertIn('time', [i['slot'] for i in plan['issues']])

    def test_unknown_city_is_not_replaced_by_selected_region(self):
        plan = self.parse('只看深圳的高风险样本')
        self.assertIn('region', [i['slot'] for i in plan['issues']])

    def test_available_named_region_is_resolved(self):
        plan = self.parse('武汉区域变化像素前3名')
        self.assertEqual(plan['region_id'], 'whu')

    def test_pixel_to_ground_area_requires_resolution_data(self):
        plan = self.parse('面积大于100平方米的前3个样本')
        self.assertIn('unit', [i['slot'] for i in plan['issues']])

    def test_knowledge_question_does_not_need_region(self):
        plan = workflow.parse_task('什么是连通域？', {}, self.regions)
        self.assertEqual(plan['mode'], 'knowledge')
        self.assertEqual(plan['issues'], [])

    def test_model_plan_cannot_choose_unregistered_tools(self):
        plan = workflow.parse_task('请分析这个', self.payload, self.regions,
            planner=lambda _: json.dumps({'mode': 'data', 'tool': 'delete_database'}))
        self.assertTrue(plan.get('plannerError'))
        self.assertNotEqual(plan.get('plannerSource'), 'model')

    def test_model_cannot_drop_a_data_task_into_knowledge_mode(self):
        plan = workflow.parse_task('最近一个月筛选建筑样本', self.payload, self.regions,
            planner=lambda _: json.dumps({'mode': 'knowledge', 'metric': None, 'tools': []}))
        self.assertIn(plan['mode'], {'data', 'mixed'})
        self.assertIn('time', [i['slot'] for i in plan['issues']])

    def test_yesterday_and_unlisted_city_are_not_ignored(self):
        self.assertIn('time', [i['slot'] for i in self.parse('昨天高风险样本前3名')['issues']])
        self.assertIn('region', [i['slot'] for i in self.parse('苏州高风险样本前3名')['issues']])

    def test_unsupported_numeric_forms_and_or_logic_need_clarification(self):
        for q in ['对象数量大于三的样本', '变化像素大于1000万的样本', '对象数量大于2或变化像素大于950的样本']:
            with self.subTest(q=q):
                self.assertTrue(self.parse(q)['issues'])


@unittest.skipUnless(importlib.util.find_spec('langgraph'), 'optional Agent runtime is not installed')
class WorkflowIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(workflow)
        import serve_change_agent_api as api
        self.api = api
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / 'data.json'
        self.data.write_text(json.dumps({'regions': [{'id': 'test', 'short': '测试区域', 'items': [
            {'id': 'A', 'changedPx': 1000, 'objectCount': 2, 'reviewRequired': False},
            {'id': 'B', 'changedPx': 900, 'objectCount': 4, 'reviewRequired': True, 'reviewLevel': 'high'},
            {'id': 'C', 'changedPx': 800, 'objectCount': 3, 'reviewRequired': True, 'reviewLevel': 'medium'},
        ]}]}), encoding='utf-8')
        api.ARGS = SimpleNamespace(use_deepseek=False, use_qwen=False, static_sql=True,
            web_data=str(self.data), knowledge_dir=str(ROOT / 'knowledge'), mysql_host=None,
            mysql_user=None, enable_rule_answer=True, enable_local_intent_corpus=False,
            model_id='none', max_new_tokens=256, prompt_version='test')
        self.calls = []
        self.requests = []
        self.checkpoint = Path(self.temp.name) / 'checkpoints.sqlite'
        self.agent = self.make_agent()
        self.addCleanup(lambda: self.agent.close())

    def make_agent(self):
        from inspection_agent import explain
        from inspection_knowledge import KnowledgeIndex
        index = KnowledgeIndex(ROOT / 'knowledge')
        def query(payload):
            self.calls.append(payload['question'])
            self.requests.append(payload)
            if payload.get('_workflowMemory'):
                self.api.db_save_session_context(self.api.db_session_key(payload), payload['_workflowMemory'])
            return self.api.sql_agent_run(payload)
        return workflow.InspectionWorkflow(query=query, retrieve=index.search,
            explain=lambda q, hits, facts: explain(q, hits, facts),
            session_key=self.api.db_session_key,
            regions=lambda: [{'id': 'test', 'short': '测试区域'}, {'id': 'whu', 'short': '武汉'}],
            checkpoint_path=self.checkpoint)

    def ask(self, q, session='one', **extra):
        return self.agent.run({'question': q, 'region': {'id': 'test'},
            'sessionKey': session, 'transientSession': True, **extra})

    def test_clarification_interrupts_before_tools_then_resumes(self):
        first = self.ask('哪些要优先看？')
        self.assertTrue(first['needsClarification'])
        self.assertEqual(self.calls, [])
        result = self.ask('按变化像素最多，给我三个')
        self.assertFalse(result.get('needsClarification', False))
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['A', 'B', 'C'])
        self.assertEqual(result['agentRuntime'], 'langgraph')

    def test_time_constraint_only_removed_after_explicit_confirmation(self):
        first = self.ask('最近一个月高风险样本前3名')
        self.assertTrue(first['needsClarification'])
        self.assertEqual(self.calls, [])
        result = self.ask('不限制时间，按当前全部样本')
        self.assertFalse(result.get('needsClarification', False))
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['B'])

    def test_sessions_do_not_resume_each_other(self):
        self.ask('哪些要优先看？', session='one')
        other = self.ask('什么是连通域？', session='two')
        self.assertFalse(other.get('needsClarification', False))
        self.assertEqual(other['answerScope'], 'knowledge')
        self.assertEqual(self.calls, [])

    def test_fresh_question_can_replace_pending_task(self):
        self.ask('哪些要优先看？')
        result = self.ask('什么是连通域？')
        self.assertEqual(result['answerScope'], 'knowledge')
        self.assertTrue(result['citations'])
        self.assertEqual(self.calls, [])

    def test_ordinal_followup_preserves_query_result_list(self):
        self.ask('当前区域变化像素前3名')
        result = self.ask('第二张为什么需要复核？', sample={'id': 'C'})
        self.assertEqual(result['evidenceSamples'][0]['sample_id'], 'B')
        self.assertTrue(result['citations'])

    def test_checkpoint_survives_workflow_restart(self):
        self.ask('哪些要优先看？')
        self.agent.close()
        self.agent = self.make_agent()
        result = self.ask('按变化像素，三个')
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['A', 'B', 'C'])

    def test_mixed_task_executes_query_and_retrieval(self):
        result = self.ask('当前区域变化像素前2名待复核样本，并解释复核依据')
        self.assertEqual([r['sample_id'] for r in result['evidenceSamples']], ['B', 'C'])
        self.assertTrue(result['citations'])
        self.assertIn('query_samples', [x['tool'] for x in result['toolTrace']])
        self.assertIn('search_knowledge', [x['tool'] for x in result['toolTrace']])

    def test_actor_namespaces_are_isolated(self):
        self.ask('哪些要优先看？', _authenticatedUser='alice')
        result = self.ask('什么是连通域？', _authenticatedUser='bob')
        self.assertFalse(result.get('needsClarification', False))

    def test_risk_order_keeps_numeric_filter(self):
        result = self.ask('按风险排序，对象数量大于3的前3个待复核样本')
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['B'])

    def test_cancelling_pending_task_does_not_execute_a_tool(self):
        self.ask('哪些要优先看？')
        result = self.ask('取消')
        self.assertEqual(result['taskPlan']['mode'], 'cancelled')
        self.assertEqual(self.calls, [])

    def test_negative_review_filter_is_preserved(self):
        result = self.ask('当前区域无需复核的样本变化像素前3名')
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['A'])

    def test_provided_region_choice_resumes_original_task(self):
        self.ask('只看深圳的高风险样本前3名')
        result = self.ask('武汉')
        self.assertFalse(result.get('needsClarification', False))
        self.assertEqual(result['taskPlan']['region_id'], 'whu')
        self.assertEqual(len(self.calls), 1)

    def test_latest_selected_sample_is_available_to_reference_clarification(self):
        self.ask('第二张为什么需要复核？')
        result = self.ask('使用当前样本', sample={'id': 'A'})
        self.assertFalse(result.get('needsClarification', False))
        self.assertEqual(result['evidenceSamples'][0]['sample_id'], 'A')

    def test_explicit_sort_metric_does_not_become_filter_metric(self):
        result = self.ask('变化像素大于850的样本按对象数量排序，前2名')
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['B', 'A'])

    def test_smallest_request_is_sorted_ascending(self):
        result = self.ask('当前区域对象数量最少的前2个样本')
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['A', 'C'])

    def test_ordinal_metric_question_keeps_original_list(self):
        self.ask('当前区域变化像素前3名')
        result = self.ask('第二张的对象数量是多少？')
        self.assertEqual(result['evidenceSamples'][0]['sample_id'], 'B')

    def test_compound_negative_risk_and_positive_review_both_apply(self):
        result = self.ask('当前区域非高风险且待复核的样本前3名')
        self.assertEqual([r['sample_id'] for r in result['sqlRows']], ['C'])

    def test_data_only_task_does_not_reuse_previous_retrieval_metadata(self):
        self.ask('什么是连通域？')
        result = self.ask('当前区域变化像素前3名')
        self.assertEqual(result['retrieval']['method'], 'not_used')


if __name__ == '__main__':
    unittest.main()
