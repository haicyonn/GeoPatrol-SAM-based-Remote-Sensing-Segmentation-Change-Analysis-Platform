import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from model_sql import PublicQueryData, ModelSQL, QueryError


def fixture(path):
    path.write_text(json.dumps({'regions': [
        {'id': 'levir', 'short': 'LEVIR', 'items': [
            {'id': 'A', 'changedPx': 200, 'objectCount': 4, 'reviewLevel': 'high', 'reviewRequired': True,
             'maskQuality': {'hole_ratio': .3}, 'reviewReasonCodes': ['large_hole_ratio']},
            {'id': 'B', 'changedPx': 0, 'objectCount': 0, 'reviewLevel': 'low', 'reviewRequired': False},
            {'id': 'C', 'objectCount': 2, 'reviewLevel': 'medium', 'reviewRequired': True}]},
        {'id': 'whu', 'short': 'WHU', 'items': [
            {'id': 'D', 'changedPx': 600, 'objectCount': 2, 'reviewLevel': 'high', 'reviewRequired': True}]}]}), encoding='utf-8')


class PublicQueryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'data.json'; fixture(self.path)
        self.data = PublicQueryData(self.path, reviewed_ids={'A'})

    def test_real_aggregation_join_and_cte(self):
        r = self.data.execute('WITH x AS (SELECT region_id, COUNT(*) n FROM samples GROUP BY region_id) '
                              'SELECT r.region_id, x.n FROM regions r JOIN x ON x.region_id=r.region_id ORDER BY r.region_id')
        self.assertEqual(r['rows'], [{'region_id': 'levir', 'n': 3}, {'region_id': 'whu', 'n': 1}])

    def test_missing_values_stay_null_and_zero_is_queryable(self):
        r = self.data.execute('SELECT sample_id FROM samples WHERE changed_px IS NULL')
        self.assertEqual(r['rows'], [{'sample_id': 'C'}])
        self.assertEqual(self.data.execute('SELECT sample_id FROM samples WHERE changed_px=0')['rows'], [{'sample_id': 'B'}])

    def test_scope_and_real_local_confirmation_are_enforced_by_the_database(self):
        r = self.data.execute('SELECT sample_id FROM samples ORDER BY sample_id', region_id='levir')
        self.assertEqual([x['sample_id'] for x in r['rows']], ['A', 'B', 'C'])
        self.assertEqual(self.data.execute('SELECT sample_id FROM samples WHERE reviewed=1')['rows'], [{'sample_id': 'A'}])

    def test_reject_writes_multiple_statements_private_tables_and_functions(self):
        bad = ["DELETE FROM samples", "SELECT 1; SELECT 2", "PRAGMA database_list",
               "SELECT * FROM sqlite_master", "SELECT * FROM samples, auth_users",
               "WITH x AS (SELECT * FROM auth_sessions) SELECT * FROM x",
               "SELECT load_extension('anything')", "SELECT randomblob(100000000)",
               "ATTACH DATABASE 'other.sqlite' AS x", "SELECT * FROM main.samples",
               "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT * FROM x"]
        for sql in bad:
            with self.subTest(sql=sql), self.assertRaises(QueryError): self.data.execute(sql)
        self.assertEqual(self.data.execute('SELECT COUNT(*) n FROM samples')['rows'][0]['n'], 4)

    def test_return_limit_and_computation_budget(self):
        r = self.data.execute('SELECT sample_id FROM samples ORDER BY sample_id', max_rows=2)
        self.assertTrue(r['truncated']); self.assertEqual(len(r['rows']), 2)
        small = PublicQueryData(self.path, step_budget=1000)
        with self.assertRaises(QueryError):
            small.execute('SELECT COUNT(*) n FROM samples a CROSS JOIN samples b CROSS JOIN samples c '
                          'CROSS JOIN samples d CROSS JOIN samples e CROSS JOIN samples f CROSS JOIN samples g')

    def test_expanded_text_cannot_fill_result_or_history(self):
        huge = "REPLACE(REPLACE(REPLACE(sample_id,'A','" + 'A'*90 + "'),'A','" + 'A'*90 + "'),'A','" + 'A'*90 + "')"
        with self.assertRaisesRegex(QueryError, '结果'):
            self.data.execute('SELECT '+huge+' AS expanded FROM samples')
        with self.assertRaisesRegex(QueryError, '结果'):
            self.data.execute('SELECT '+','.join("REPLACE(REPLACE(sample_id,'A','"+'A'*40+"'),'A','"+'A'*40+"') AS c"+str(i) for i in range(20))+' FROM samples')

    def test_human_verdict_is_queryable_separately_from_prediction_risk(self):
        raw=json.loads(self.path.read_text());items=raw['regions'][0]['items']
        for sample,verdict in zip(items,('confirmed_change','false_change','uncertain')):
            sample['manualReviewComplete']=True;sample['manualReview']={'verdict':verdict,'version':1}
        self.path.write_text(json.dumps(raw))
        data=PublicQueryData(self.path)
        self.assertEqual(data.execute("SELECT sample_id,review_verdict FROM samples WHERE review_verdict='false_change'")['rows'],[{'sample_id':'B','review_verdict':'false_change'}])
        self.assertEqual(data.execute("SELECT COUNT(*) AS n FROM samples WHERE reviewed=1")['rows'],[{'n':3}])
        self.assertEqual(data.execute("SELECT review_verdict FROM samples WHERE sample_id='D'")['rows'],[{'review_verdict':None}])


class ModelSQLTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'data.json'; fixture(self.path)
        self.history = Path(self.temp.name) / 'history.sqlite'
        self.prompts = []

    def service(self, outputs):
        iterator = iter(outputs)
        def generate(prompt, **kwargs):
            self.prompts.append(prompt)
            return {'text': next(iterator), 'finishReason': 'stop', 'outputTokens': 30}
        return ModelSQL(self.path, self.history, generate=generate, model_id='real-test-model')

    def payload(self, q, session='one', user='alice'):
        return {'question': q, 'sessionKey': session, '_authenticatedUser': user}

    def test_sql_is_the_model_output_and_not_a_question_template(self):
        s = self.service(['{"sql":"SELECT SUM(object_count) AS sum_objects FROM samples","clarification":null}'])
        r = s.run(self.payload('把四个样本的对象数量加起来'))
        self.assertEqual(r['status'], 'completed')
        self.assertEqual(r['sqlRows'], [{'sum_objects': 8}])
        self.assertEqual(r['sqlPlanSource'], 'model'); self.assertEqual(r['llmCalls'], 1)
        self.assertEqual(r['modelId'], 'real-test-model')
        self.assertIn('CREATE TABLE', self.prompts[0])

    def test_execution_error_is_passed_back_for_one_real_repair(self):
        s = self.service(['SELECT not_a_column FROM samples', 'SELECT COUNT(*) n FROM samples'])
        r = s.run(self.payload('有多少样本'))
        self.assertEqual(r['status'], 'completed'); self.assertEqual(r['llmCalls'], 2)
        self.assertIn('not_a_column', self.prompts[1])
        self.assertEqual(r['sqlRows'], [{'n': 4}])
        self.assertEqual(len(r['attempts']), 2)
        self.assertNotIn('error', r)

    def test_unsafe_output_and_failed_repairs_never_fall_back_to_templates(self):
        s = self.service(['DELETE FROM samples', 'SELECT password_hash FROM auth_users'])
        r = s.run(self.payload('给我样本'))
        self.assertEqual(r['status'], 'failed'); self.assertEqual(r['sqlRows'], [])
        self.assertEqual(r['llmCalls'], 2)

    def test_clarification_does_not_execute_a_query(self):
        s = self.service(['{"sql":null,"clarification":"请说明按变化面积还是对象数量排序。"}'])
        r = s.run(self.payload('哪个最重要'))
        self.assertEqual(r['status'], 'needs_clarification')
        self.assertEqual(r['sqlRows'], [])

    def test_clarification_reply_keeps_the_pending_question(self):
        s = self.service(['{"sql":null,"clarification":"按什么排序？"}',
                          'SELECT sample_id FROM samples ORDER BY changed_px DESC LIMIT 1'])
        s.run(self.payload('给我最突出的样本'))
        s.run(self.payload('按变化像素'))
        self.assertIn('给我最突出的样本', self.prompts[1])

    def test_multiple_clarifications_keep_original_and_all_replies(self):
        s = self.service(['{"sql":null,"clarification":"按什么排序？"}',
                          '{"sql":null,"clarification":"哪个风险级别？"}', 'SELECT sample_id FROM samples'])
        s.run(self.payload('LEVIR变化像素大于123的前5个样本'))
        s.run(self.payload('按对象数降序'))
        s.run(self.payload('只要高风险'))
        self.assertIn('LEVIR变化像素大于123的前5个样本', self.prompts[2])
        self.assertIn('按对象数降序', self.prompts[2])

    def test_review_change_during_generation_does_not_return_stale_data(self):
        reviewed = set()
        def generate(*a, **kw):
            reviewed.add('A')
            return 'SELECT COUNT(*) n FROM samples WHERE reviewed=1'
        s = ModelSQL(self.path, self.history, generate=generate, model_id='test', reviewed_provider=lambda:reviewed)
        r = s.run(self.payload('多少已复核'))
        self.assertEqual(r['status'], 'failed')
        self.assertEqual(r['sqlRows'], [])
        self.assertEqual(r['llmCalls'], 1)
        self.assertIn('复核', r['error'])

    def test_review_version_changes_and_invalidates_history(self):
        s = self.service(['SELECT sample_id FROM samples WHERE reviewed=0', 'SELECT COUNT(*) n FROM samples WHERE reviewed=1'])
        reviewed=set();s.reviewed_provider=lambda:reviewed
        a = s.run(self.payload('旧复核状态专用条件'))
        reviewed.add('A')
        b = s.run(self.payload('多少已复核'))
        self.assertEqual(a['sourceVersion'],b['sourceVersion'])
        self.assertNotEqual(a['dataVersion'],b['dataVersion'])
        self.assertEqual(b['sqlRows'],[{'n':1}])
        self.assertNotIn('旧复核状态专用条件',self.prompts[1])

    def test_context_limit_is_explicit_and_not_silent_condition_loss(self):
        s = self.service(['{"sql":null,"clarification":"请补充。"}']*8)
        statuses=[]
        for _ in range(8):
            r=s.run(self.payload('原条件'*1000));statuses.append(r['status'])
            if r['status']=='failed':break
        self.assertIn('failed',statuses)
        self.assertIn('上下文',r['error'])
        self.assertLessEqual(max(map(len,self.prompts)),16000)

    def test_rejected_prompt_does_not_count_a_model_call(self):
        s = self.service(['SELECT COUNT(*) n FROM samples']*2)
        r = s.run(self.payload('\x00'*4000))
        self.assertEqual(r['llmCalls'],len(self.prompts))
        self.assertEqual(r['llmCalls'],0)

    def test_previous_actual_results_are_in_context_but_other_users_are_not(self):
        s = self.service(['SELECT sample_id FROM samples ORDER BY sample_id LIMIT 2',
                          "SELECT sample_id FROM samples WHERE sample_id='B'", 'SELECT COUNT(*) n FROM samples'])
        s.run(self.payload('我第一轮独有的问题'))
        r = s.run(self.payload('第二个呢'))
        self.assertEqual(r['sqlRows'], [{'sample_id': 'B'}])
        self.assertIn('我第一轮独有的问题', self.prompts[1])
        self.assertIn('"sample_id": "B"', self.prompts[1])
        s.run(self.payload('有多少条', user='bob'))
        self.assertNotIn('我第一轮独有的问题', self.prompts[2])

    def test_source_change_discards_old_history(self):
        s = self.service(['SELECT COUNT(*) n FROM samples', 'SELECT COUNT(*) n FROM samples'])
        s.run(self.payload('旧来源专用问题'))
        raw = json.loads(self.path.read_text(encoding='utf-8')); raw['regions'][0]['items'].append({'id': 'E'})
        self.path.write_text(json.dumps(raw), encoding='utf-8')
        r = s.run(self.payload('现在多少条'))
        self.assertEqual(r['sqlRows'], [{'n': 5}])
        self.assertNotIn('旧来源专用问题', self.prompts[1])

    def test_truncated_output_is_not_executed(self):
        def generate(*a, **kw): return {'text': 'SELECT COUNT(*) n FROM samples', 'finishReason': 'length'}
        s = ModelSQL(self.path, self.history, generate=generate, model_id='test')
        self.assertEqual(s.run(self.payload('多少条'))['status'], 'failed')

    def test_model_unavailable_has_no_fake_sql(self):
        s = ModelSQL(self.path, self.history, generate=None, model_id='disabled')
        r = s.run(self.payload('多少条'))
        self.assertEqual(r['status'], 'model_unavailable'); self.assertEqual(r['llmCalls'], 0)


if __name__ == '__main__': unittest.main()
