import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
try:
    module = importlib.import_module('analysis_tasks')
except ModuleNotFoundError:
    module = None


class AnalysisTaskTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(module, 'The task analysis engine is not implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'facts.json'
        self.source.write_text(json.dumps({'regions': [
            {'id': 'whu', 'short': 'WHU', 'items': [
                {'id': 'A', 'stem': 'A', 'changedPx': 10, 'objectCount': 1},
                {'id': 'B', 'stem': 'B', 'changedPx': 30, 'objectCount': 3}]},
            {'id': 'levir', 'short': 'LEVIR', 'items': [
                {'id': 'C', 'stem': 'C', 'changedPx': 900, 'objectCount': 4}]}]}, ensure_ascii=False), encoding='utf-8')
        self.outputs = []
        self.prompts = []
        self.reviewed = set()
        def generate(prompt, **kwargs):
            self.prompts.append(prompt)
            return {'text': json.dumps(self.outputs.pop(0), ensure_ascii=False),
                    'inputTokens': 100, 'outputTokens': 30, 'finishReason': 'stop'}
        self.service = module.AnalysisTasks(self.source, Path(self.temp.name) / 'tasks.sqlite',
            generate=generate, model_id='test-model', reviewed_provider=lambda: self.reviewed)

    def call(self, action, **kwargs):
        return self.service.handle(dict(action=action, sessionKey='test-session', **kwargs))

    def plan(self, **kwargs):
        self.outputs.append({'title': '区域变化分析', 'steps': [
            {'title': '变化量概况', 'question': 'WHU样本数量与平均变化像素',
             'sql': 'SELECT COUNT(*) AS n,AVG(changed_px) AS mean FROM samples'}], 'clarification': None})
        return self.call('plan', goal='分析WHU的样本数量与平均变化像素', **kwargs)

    def test_plan_does_not_execute_and_execution_physically_limits_region(self):
        plan = self.plan()
        self.assertEqual(plan['status'], 'planned')
        self.assertNotIn('rows', plan['steps'][0])
        self.assertEqual(plan['scopeRegion'], 'whu')
        self.outputs.append({'findings': [{'text': '样本数量为2，平均变化像素为20。', 'evidence': ['S1:R1']}], 'nextQuestions': []})
        result = self.call('execute', planId=plan['planId'])
        self.assertEqual(result['steps'][0]['rows'], [{'n': 2, 'mean': 20.0}])
        self.assertEqual(result['report']['status'], 'generated')
        self.assertEqual(result['evidence'][0]['id'], 'S1:R1')

    def test_client_sql_cannot_override_stored_plan_and_repeat_is_idempotent(self):
        plan = self.plan()
        self.outputs.append({'findings': [], 'nextQuestions': []})
        first = self.call('execute', planId=plan['planId'], steps=[{'sql': 'DROP TABLE samples'}])
        calls = len(self.prompts)
        second = self.call('execute', planId=plan['planId'])
        self.assertEqual(first, second)
        self.assertEqual(calls, len(self.prompts))
        self.assertEqual(first['steps'][0]['rows'][0]['n'], 2)

    def test_owner_isolation(self):
        plan = self.plan()
        with self.assertRaises(ValueError):
            self.service.handle(dict(action='execute',sessionKey='other',planId=plan['planId']))

    def test_source_change_and_review_change_invalidate_plan(self):
        plan = self.plan()
        self.source.write_text(self.source.read_text(encoding='utf-8') + ' ', encoding='utf-8')
        with self.assertRaises(ValueError): self.call('execute', planId=plan['planId'])
        plan = self.plan()
        self.reviewed.add('A')
        with self.assertRaises(ValueError): self.call('execute', planId=plan['planId'])

    def test_mid_execution_version_change_is_not_reported_as_complete(self):
        plan = self.plan()
        old = self.service.generate
        def change(prompt, **kwargs):
            self.reviewed.add('A')
            return old(prompt, **kwargs)
        self.service.generate = change
        self.outputs.append({'findings': [], 'nextQuestions': []})
        with self.assertRaises(ValueError): self.call('execute', planId=plan['planId'])

    def test_cross_region_request_explains_scope_change_before_execution(self):
        self.outputs.append({'title':'区域比较','clarification':None,'steps':[{'title':'数量',
            'question':'WHU和LEVIR各自样本数量',
            'sql':'SELECT region_id,COUNT(*) AS n FROM samples GROUP BY region_id'}]})
        plan=self.call('plan',goal='比较WHU和LEVIR的变化分布',scopeRegion='whu')
        self.assertIsNone(plan['scopeRegion'])
        self.assertEqual(plan['targetRegion'],'whu')
        self.assertIn('比较',plan['scopeAdjustment'])
        self.assertEqual(plan['status'],'planned')
        self.assertNotIn('rows',plan['steps'][0])

    def comparison_plan(self, goal='分析当前区域的变化率，对比第三高的区域看看高多少', **kwargs):
        raw=json.loads(self.source.read_text(encoding='utf-8'))
        raw['regions'][0]['items'][0]['changedRatio']=.8
        raw['regions'][0]['items'][1]['changedRatio']=.6
        raw['regions'][1]['items'][0]['changedRatio']=.2
        raw['regions'].append({'id':'uav','short':'UAV','items':[{'id':'D','changedRatio':.4}]})
        raw['regions'].append({'id':'gzcd','short':'GZCD','items':[{'id':'E','changedRatio':.3}]})
        self.source.write_text(json.dumps(raw),encoding='utf-8')
        rank='SELECT region_id,AVG(changed_ratio) AS rate FROM samples WHERE changed_ratio IS NOT NULL GROUP BY region_id ORDER BY rate DESC LIMIT 1 OFFSET 2'
        compare="WITH target AS (SELECT AVG(changed_ratio) AS rate FROM samples WHERE region_id='levir'), baseline AS ("+rank+") SELECT baseline.region_id,target.rate AS current_rate,baseline.rate AS baseline_rate,(target.rate-baseline.rate)*100 AS difference_pp FROM target CROSS JOIN baseline"
        self.outputs.append({'title':'变化率比较','clarification':None,'steps':[
            {'title':'当前值','question':'LEVIR变化率均值','sql':"SELECT AVG(changed_ratio) AS rate FROM samples WHERE region_id='levir'"},
            {'title':'第三名','question':'全部区域变化率均值第三高的区域','sql':rank},
            {'title':'比较','question':'LEVIR与全部区域第三高区域的均值差，单位百分点','sql':compare}]})
        return self.call('plan',goal=goal,**kwargs)

    def test_region_rank_keeps_target_separate_from_candidate_scope(self):
        plan=self.comparison_plan(scopeRegion='levir')
        self.assertIsNone(plan['scopeRegion'])
        self.assertEqual(plan['targetRegion'],'levir')
        self.assertIn('levir',self.prompts[0])
        self.outputs.append({'findings':[],'nextQuestions':[]})
        result=self.call('execute',planId=plan['planId'])
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['steps'][1]['rows'],[{'region_id':'gzcd','rate':.3}])
        self.assertAlmostEqual(result['steps'][2]['rows'][0]['difference_pp'],-10)

    def test_one_named_target_does_not_reduce_region_ranking_candidates(self):
        plan=self.comparison_plan(goal='LEVIR的变化率和排名第三的区域比相差多少')
        self.assertIsNone(plan['scopeRegion'])
        self.assertEqual(plan['targetRegion'],'levir')

    def test_current_region_without_anchor_is_clarified_before_model(self):
        result=self.call('plan',goal='当前区域的变化率和第三高的区域比相差多少')
        self.assertEqual(result['status'],'needs_clarification')
        self.assertEqual(self.prompts,[])

    def test_unavailable_region_rank_is_explained_before_model(self):
        result=self.call('plan',goal='按各区域样本数量降序排名，查询第3名区域及其样本数量')
        self.assertEqual(result['status'],'needs_clarification')
        self.assertIn('只有2个',result['answer'])
        self.assertEqual(self.prompts,[])

    def test_unavailable_chinese_region_rank_is_explained_before_model(self):
        result=self.call('plan',goal='查询第三高的区域变化比例')
        self.assertEqual(result['status'],'needs_clarification')
        self.assertIn('不存在第3名',result['answer'])
        self.assertEqual(self.prompts,[])

    def test_current_target_is_not_replaced_by_named_comparison_region(self):
        self.outputs.append({'title':'比较','clarification':None,'steps':[{'title':'比较',
            'question':'LEVIR与WHU样本数量','sql':'SELECT region_id,COUNT(*) AS n FROM samples GROUP BY region_id'}]})
        plan=self.call('plan',goal='当前区域与WHU比较',scopeRegion='levir')
        self.assertEqual(plan['targetRegion'],'levir')
        self.assertIsNone(plan['scopeRegion'])

    def test_zero_count_is_a_valid_result(self):
        self.assertIsNone(module.result_gap('SELECT COUNT(*) AS n FROM samples WHERE 1=0',[{'n':0}]))

    def test_regional_rank_cannot_hide_the_candidate_identifier(self):
        with self.assertRaisesRegex(ValueError,'区域标识'):
            module.validate_step('SELECT AVG(changed_ratio) AS rate FROM samples GROUP BY region_id ORDER BY rate DESC LIMIT 1 OFFSET 2',
                                 '第三高区域的变化率',[],None,'与第三高的区域比较')

    def test_rank_repair_cannot_reintroduce_target_filter_into_candidates(self):
        regions=[{'id':'levir'},{'id':'whu'},{'id':'uav'}]
        for predicate in ["region_id='levir'", "region_id IN (SELECT region_id FROM regions WHERE region_name='levir')"]:
            sql='WITH ranked AS (SELECT region_id,AVG(changed_ratio) AS rate FROM samples WHERE '+predicate+' GROUP BY region_id) SELECT region_id,rate FROM ranked ORDER BY rate DESC LIMIT 1 OFFSET 2'
            with self.subTest(predicate=predicate),self.assertRaisesRegex(ValueError,'候选数据'):
                module.validate_step(sql,'第三高区域的变化率',regions,None,'LEVIR和排名第三的区域比相差多少')

    def test_plan_missing_requested_difference_is_repaired_before_confirmation(self):
        incomplete={'title':'比较','clarification':None,'steps':[{'title':'数量','question':'各区域数量',
            'sql':'SELECT region_id,COUNT(*) AS n FROM samples GROUP BY region_id'}]}
        self.outputs.append(incomplete)
        self.outputs.append({'title':'差值比较','clarification':None,'steps':[{'title':'差值','question':'区域数量差',
            'sql':"SELECT (SELECT COUNT(*) FROM samples WHERE region_id='whu')-(SELECT COUNT(*) FROM samples WHERE region_id='levir') AS difference"}]})
        plan=self.call('plan',goal='WHU和第二高的区域样本数相差多少')
        self.assertEqual(plan['status'],'planned')
        self.assertEqual([entry['stage'] for entry in plan['modelUsage']],['plan','plan_repair'])

    def test_repeatedly_incomplete_comparison_plan_is_not_accepted(self):
        incomplete={'title':'比较','clarification':None,'steps':[{'title':'数量','question':'各区域数量',
            'sql':'SELECT region_id,COUNT(*) AS n FROM samples GROUP BY region_id'}]}
        self.outputs.extend([incomplete,incomplete])
        with self.assertRaisesRegex(ValueError,'仍未包含'):
            self.call('plan',goal='WHU和第二高的区域样本数相差多少')

    def test_unit_inference_does_not_trust_the_alias(self):
        sql="WITH a AS (SELECT AVG(changed_ratio) AS current_ratio FROM samples WHERE region_id='levir'), b AS (SELECT AVG(changed_ratio) AS baseline_ratio FROM samples) SELECT (current_ratio-baseline_ratio)*100 AS misleading_pct FROM a CROSS JOIN b"
        self.assertEqual(module.measure_units(sql),{'misleading_pct':'percentage_points'})

    def test_min_max_ratio_keeps_units_for_rounded_percent_report(self):
        sql='SELECT AVG(foreground_ratio) AS mean, MIN(foreground_ratio) AS lo, MAX(foreground_ratio) AS hi FROM samples'
        units=module.measure_units(sql)
        self.assertEqual(units,{'mean':'ratio','lo':'ratio','hi':'ratio'})
        evidence=[{'id':'S1:R1','values':{'mean':.15879047393798829,'lo':0.,'hi':.4082450866699219},'units':units}]
        self.assertTrue(module.check_findings({'findings':[{'text':'平均建筑占比15.88%，最低0%，最高40.82%。','evidence':['S1:R1']}]},evidence))
        with self.assertRaises(ValueError):
            module.check_findings({'findings':[{'text':'最高建筑占比40.83%。','evidence':['S1:R1']}]},evidence)

    def test_report_accepts_rounding_and_signed_point_difference_but_not_percent(self):
        evidence=[{'id':'S1:R1','values':{'rate':.05043202978515625,'difference_pct':-1.72766764648},
                   'units':{'rate':'ratio','difference_pct':'percentage_points'}}]
        good={'findings':[{'text':'变化比例为0.0504，相比基准低1.73个百分点。','evidence':['S1:R1']}]}
        self.assertTrue(module.check_findings(good,evidence))
        with self.assertRaisesRegex(ValueError,'百分点'):
            module.check_findings({'findings':[{'text':'相比基准低1.73%。','evidence':['S1:R1']}]},evidence)

    def test_report_accepts_only_verified_ratio_to_percent_conversion(self):
        evidence=[{'id':'S1:R1','values':{'avg_changed_ratio':.13844085041666665},'units':{'avg_changed_ratio':'ratio'}}]
        for text in ['UAV变化率为13.84%。','UAV变化率为百分之十三点八四。','UAV变化比例为0.138441。']:
            with self.subTest(text=text):
                self.assertTrue(module.check_findings({'findings':[{'text':text,'evidence':['S1:R1']}]},evidence))
        for text in ['UAV变化率为13.85%。','UAV变化率为0.14%。','UAV变化率为1384.41%。','UAV变化比例为13.84。']:
            with self.subTest(text=text),self.assertRaises(ValueError):
                module.check_findings({'findings':[{'text':text,'evidence':['S1:R1']}]},evidence)

    def test_percent_conversion_is_bound_to_its_cited_row(self):
        evidence=[{'id':'S1:R1','values':{'rate':.06770870625},'units':{'rate':'ratio'}},
                  {'id':'S1:R2','values':{'rate':.13844085041666665},'units':{'rate':'ratio'}}]
        with self.assertRaises(ValueError):
            module.check_findings({'findings':[{'text':'UAV变化率为13.84%。','evidence':['S1:R1']}]},evidence)

    def test_percent_values_are_not_multiplied_twice_and_points_remain_distinct(self):
        evidence=[{'id':'S1:R1','values':{'share':23.599670116257414},'units':{'share':'percent'}}]
        self.assertTrue(module.check_findings({'findings':[{'text':'贡献占比为23.60%。','evidence':['S1:R1']}]},evidence))
        with self.assertRaises(ValueError):
            module.check_findings({'findings':[{'text':'贡献占比为2359.97%。','evidence':['S1:R1']}]},evidence)

    def test_ratio_difference_converts_to_points_only(self):
        evidence=[{'id':'S1:R1','values':{'difference':-.01727667646484375},'units':{'difference':'ratio_difference'}}]
        self.assertTrue(module.check_findings({'findings':[{'text':'当前区域低1.73个百分点。','evidence':['S1:R1']}]},evidence))
        for text in ['当前区域低1.73%。','差值为-0.0173个百分点。']:
            with self.subTest(text=text),self.assertRaises(ValueError):
                module.check_findings({'findings':[{'text':text,'evidence':['S1:R1']}]},evidence)

    def test_model_display_metadata_cannot_authorize_a_fake_number(self):
        evidence=[{'id':'S1:R1','values':{'rate':.13844085041666665},'units':{'rate':'ratio'},
                   'displayValues':[{'value':'99','unit':'percent'}]}]
        with self.assertRaises(ValueError):
            module.check_findings({'findings':[{'text':'变化率为99%。','evidence':['S1:R1']}]},evidence)

    def test_report_repair_is_bounded_and_keeps_invalid_candidates(self):
        plan=self.plan()
        self.outputs.extend([{'findings':[{'text':'数量为999。','evidence':['S1:R1']}],'nextQuestions':[]},
                             {'findings':[{'text':'数量为888。','evidence':['S1:R1']}],'nextQuestions':[]}])
        result=self.call('execute',planId=plan['planId'])
        self.assertEqual(result['report']['status'],'rejected')
        self.assertEqual(result['report']['findings'],[])
        self.assertEqual(len(result['report']['attempts']),2)
        self.assertEqual(result['llmCalls'],3)

    def test_report_does_not_omit_a_comparison_side_and_repairs_without_rerunning_sql(self):
        raw=json.loads(self.source.read_text(encoding='utf-8'))
        for item in raw['regions'][0]['items']:item['changedRatio']=.2
        raw['regions'][1]['items'][0]['changedRatio']=.4
        self.source.write_text(json.dumps(raw),encoding='utf-8')
        self.outputs.append({'title':'两区比较','clarification':None,'steps':[{'title':'比较两区','question':'比较WHU和LEVIR的变化比例均值',
            'sql':"SELECT region_id,AVG(changed_ratio) AS rate FROM samples WHERE region_id IN ('whu','levir') GROUP BY region_id ORDER BY region_id"}]})
        plan=self.call('plan',goal='比较WHU和LEVIR的变化比例均值')
        self.outputs.extend([
            {'findings':[{'text':'WHU变化率为20%。','evidence':['S1:R2']}],'nextQuestions':[]},
            {'findings':[{'text':'LEVIR变化率为40%，WHU为20%。','evidence':['S1:R1','S1:R2']}],'nextQuestions':[]}])
        result=self.call('execute',planId=plan['planId'])
        self.assertEqual(result['report']['status'],'generated')
        self.assertEqual([x['status'] for x in result['report']['attempts']],['rejected','accepted'])
        self.assertIn('遗漏',result['report']['attempts'][0]['error'])
        self.assertEqual(len(result['steps'][0]['attempts']),1)
        self.assertEqual([m['stage'] for m in result['modelUsage']],['plan','report','report_repair'])

    def test_old_cached_plan_requires_replanning(self):
        import sqlite3
        plan=self.plan()
        with sqlite3.connect(self.service.history_path) as db:
            old=dict(plan);old.pop('analysisVersion')
            db.execute('UPDATE analysis_runs SET plan=?,result=? WHERE id=?',
                       (json.dumps(old),json.dumps({'status':'completed'}),plan['planId']))
        with self.assertRaisesRegex(ValueError,'旧版'):
            self.call('execute',planId=plan['planId'])

    def test_empty_rank_is_partial_even_though_sql_executed(self):
        self.outputs.append({'title':'区域排名','clarification':None,'steps':[
            {'title':'样本数','question':'各区域样本数','sql':'SELECT region_id,COUNT(*) AS n FROM samples GROUP BY region_id'},
            {'title':'第三名','question':'全部区域样本数第三高的区域','sql':'SELECT region_id,COUNT(*) AS n FROM samples GROUP BY region_id ORDER BY n DESC LIMIT 1 OFFSET 2'}]})
        plan=self.call('plan',goal='统计各区域样本数并找出第三名')
        self.outputs.append({'findings':[],'nextQuestions':[]})
        result=self.call('execute',planId=plan['planId'])
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['steps'][1]['status'],'no_data')
        self.assertEqual(result['steps'][1]['attempts'][0]['status'],'executed')
        self.assertIn('未完整完成',result['report']['message'])
        self.assertTrue(result['report']['unansweredSteps'])
        self.assertEqual(len(result['steps'][1]['attempts']),1)

    def test_null_aggregate_is_not_an_answer_but_zero_count_is(self):
        self.outputs.append({'title':'未知变化率','clarification':None,'steps':[
            {'title':'数量','question':'样本数量','sql':'SELECT COUNT(*) AS n FROM samples'},
            {'title':'变化率','question':'平均变化比例','sql':'SELECT AVG(changed_ratio) AS rate FROM samples'}]})
        plan=self.call('plan',goal='统计样本数量与变化比例均值')
        self.outputs.append({'findings':[],'nextQuestions':[]})
        result=self.call('execute',planId=plan['planId'])
        self.assertEqual(result['steps'][0]['status'],'completed')
        self.assertEqual(result['steps'][1]['status'],'no_data')
        self.assertEqual(result['status'],'partial')

    def test_missing_date_and_ground_area_are_not_silently_removed(self):
        for goal in ['分析WHU今年的变化量', '统计WHU变化面积大于10平方米的样本']:
            with self.subTest(goal=goal), self.assertRaises(ValueError): self.call('plan', goal=goal)
        self.assertEqual(self.prompts, [])

    def test_model_cannot_plan_writes_or_more_than_three_queries(self):
        for steps in [[{'title':'删除','question':'删除记录','sql':'DELETE FROM samples'}],
                      [{'title':'查询','question':'统计','sql':'SELECT COUNT(*) FROM samples'}] * 4]:
            self.outputs.append({'title':'无效方案','steps':steps,'clarification':None})
            with self.subTest(steps=len(steps)), self.assertRaises(ValueError): self.call('plan',goal='分析WHU')

    def test_sql_error_is_repaired_once_and_original_error_remains_visible(self):
        self.outputs.append({'title':'分析','steps':[{'title':'数量','question':'样本数量',
            'sql':'SELECT COUNT(missing_column) AS n FROM samples'}],'clarification':None})
        plan=self.call('plan',goal='统计WHU样本数量')
        self.outputs.extend([{'sql':'SELECT COUNT(*) AS n FROM samples'}, {'findings':[],'nextQuestions':[]}])
        r=self.call('execute',planId=plan['planId'])
        self.assertEqual(r['steps'][0]['rows'],[{'n':2}])
        self.assertEqual([x['status'] for x in r['steps'][0]['attempts']], ['rejected','executed'])

    def test_report_unknown_evidence_and_invented_numbers_are_rejected(self):
        for finding in [{'text':'数量为999。','evidence':['S1:R1']},
                        {'text':'这些样本变化量集中。','evidence':['S9:R9']},
                        {'text':'实际面积为20平方米。','evidence':['S1:R1']}]:
            plan=self.plan()
            self.outputs.append({'findings':[finding],'nextQuestions':[]})
            result=self.call('execute',planId=plan['planId'])
            self.assertEqual(result['report']['status'],'rejected')
            self.assertEqual(result['report']['findings'],[])
            self.assertTrue(result['steps'][0]['rows'])

    def test_report_cannot_hide_unsupported_numbers_in_chinese(self):
        plan=self.plan()
        self.outputs.append({'findings':[{'text':'共有九百九十九张样本。','evidence':['S1:R1']}],'nextQuestions':[]})
        self.assertEqual(self.call('execute',planId=plan['planId'])['report']['status'],'rejected')

    def test_global_denominator_is_not_physically_reduced_to_one_named_region(self):
        self.outputs.append({'title':'WHU占比','clarification':None,'steps':[{'title':'全库占比',
            'question':'WHU样本占全部区域样本总数的百分比',
            'sql':"SELECT ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM samples),2) AS share_pct FROM samples WHERE region_id='whu'"}]})
        plan=self.call('plan',goal='计算WHU样本占全部区域样本总数的百分比')
        self.outputs.append({'findings':[],'nextQuestions':[]})
        result=self.call('execute',planId=plan['planId'])
        self.assertEqual(result['steps'][0]['rows'],[{'share_pct':66.67}])

    def test_global_count_of_public_sample_id_is_equivalent_to_count_star(self):
        sql="SELECT ROUND(100.0*COUNT(sample_id)/(SELECT COUNT(sample_id) FROM samples),2) AS share_pct FROM samples WHERE region_id='whu'"
        self.outputs.append({'title':'WHU占比','clarification':None,'steps':[{'title':'全库占比',
            'question':'WHU样本占全库样本总数的百分比，保留两位小数','sql':sql}]})
        plan=self.call('plan',goal='计算WHU样本占全库样本总数的百分比')
        self.outputs.append({'findings':[],'nextQuestions':[]})
        r=self.call('execute',planId=plan['planId'])
        self.assertEqual(r['steps'][0]['rows'],[{'share_pct':66.67}])
        self.assertEqual(len(r['steps'][0]['attempts']),1)

    def test_comparison_of_review_states_is_not_repaired_into_excluding_one(self):
        self.reviewed.add('A')
        self.outputs.append({'title':'状态分布','clarification':None,'steps':[{'title':'两组数量',
            'question':'分别统计WHU已复核和未复核的数量',
            'sql':'SELECT reviewed,COUNT(*) AS n FROM samples GROUP BY reviewed ORDER BY reviewed'}]})
        plan=self.call('plan',goal='分别比较WHU已复核和未复核的数量')
        self.outputs.append({'findings':[],'nextQuestions':[]})
        result=self.call('execute',planId=plan['planId'])
        self.assertEqual(result['steps'][0]['rows'],[{'reviewed':0,'n':1},{'reviewed':1,'n':1}])
        self.assertEqual(len(result['steps'][0]['attempts']),1)

    def test_chinese_percentage_cannot_bypass_report_numeric_checks(self):
        plan=self.plan()
        self.outputs.append({'findings':[{'text':'占比为百分之九十九。','evidence':['S1:R1']}],'nextQuestions':[]})
        self.assertEqual(self.call('execute',planId=plan['planId'])['report']['status'],'rejected')

    def test_report_accepts_explicit_two_decimal_display_and_executed_query_parameters(self):
        evidence=[dict(id='S1:R1',values={'mean':2529.542395693136,'share':23.6},parameters=[10])]
        checked=module.check_findings({'findings':[{'text':'均值为2529.54，前10张占比为23.6%。','evidence':['S1:R1']}]},evidence)
        self.assertEqual(len(checked),1)

    def test_rejected_model_report_is_preserved_as_untrusted_diagnostic(self):
        plan=self.plan()
        self.outputs.append({'findings':[{'text':'共有999张样本。','evidence':['S1:R1']}],'nextQuestions':[]})
        r=self.call('execute',planId=plan['planId'])
        self.assertEqual(r['report']['status'],'rejected')
        self.assertEqual(r['report']['candidate']['findings'][0]['text'],'共有999张样本。')

    def test_report_context_is_bounded_without_silently_hiding_rows_in_tables(self):
        raw=json.loads(self.source.read_text(encoding='utf-8'))
        raw['regions'][0]['items']=[{'id':f'A{i}','changedPx':i,'objectCount':1} for i in range(50)]
        self.source.write_text(json.dumps(raw),encoding='utf-8')
        self.outputs.append({'title':'样本概况','clarification':None,'steps':[{'title':'明细','question':'样本明细',
            'sql':'SELECT sample_id,changed_px FROM samples ORDER BY sample_id'}]})
        plan=self.call('plan',goal='列出WHU样本明细并分析')
        self.outputs.append({'findings':[],'nextQuestions':[]})
        result=self.call('execute',planId=plan['planId'])
        self.assertEqual(len(result['steps'][0]['rows']),50)
        self.assertLess(len(result['reportEvidenceIds']),len(result['evidence']))
        self.assertLess(len(self.prompts[-1]),12000)


if __name__=='__main__':unittest.main()
