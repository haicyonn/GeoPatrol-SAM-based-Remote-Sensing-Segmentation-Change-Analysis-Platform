import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_sql import ModelSQL
from test_model_sql import fixture


def contract(question='查询样本',**overrides):
    value=dict(action='query',question=question,regions=['levir'],
        columns=['sample_id'],order=[],limit=None,offset=0,per_group=False,
        calculations=[],missing=None)
    value.update(overrides)
    return json.dumps(value,ensure_ascii=False)


class QueryContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'data.json';fixture(self.path)
        self.prompts=[]

    def service(self,outputs):
        sequence=iter(outputs)
        def generate(prompt,**kwargs):
            self.prompts.append(prompt)
            return {'text':next(sequence),'finishReason':'stop','inputTokens':30,'outputTokens':30}
        s=ModelSQL(self.path,Path(self.temp.name)/'queries.sqlite',generate=generate,model_id='test',contract_mode=True)
        return s

    def run_query(self,s,q,scope=None,session='one',user='alice'):
        return s.run(dict(question=q,scopeRegion=scope,sessionKey=session,_authenticatedUser=user))

    def test_computed_average_is_generated_by_model_and_executed(self):
        p=contract('统计LEVIR大于本区域均值的样本数量',columns=['n'],calculations=['AVG(changed_px) within levir'])
        sql="SELECT COUNT(*) AS n FROM samples WHERE region_id='levir' AND changed_px>(SELECT AVG(changed_px) FROM samples WHERE region_id='levir')"
        s=self.service([p,sql]);r=self.run_query(s,'LEVIR中高于本区域平均变化像素的样本有多少张？')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['sqlRows'],[{'n':1}])
        self.assertIn('queryContract',r)
        self.assertEqual(r['queryContract']['calculations'],['AVG(changed_px) within levir'])
        self.assertEqual(r['llmCalls'],2)

    def test_count_question_rejects_row_id_contract_before_sql_generation(self):
        wrong=contract('LEVIR有多少样本',columns=['sample_id'])
        right=contract('LEVIR有多少样本',columns=['sample_count'],calculations=['COUNT(*)'])
        s=self.service([wrong,right,"SELECT COUNT(*) AS sample_count FROM samples WHERE region_id='levir'"])
        r=self.run_query(s,'LEVIR有多少样本？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'sample_count':3}])
        self.assertEqual(r['attempts'][0]['status'],'rejected');self.assertIn('计数',r['attempts'][0]['error'])

    def test_extra_columns_and_missing_order_trigger_real_sql_repair(self):
        p=contract('LEVIR变化像素第2至3张',columns=['sample_id','changed_px'],
            order=[{'expression':'changed_px','direction':'DESC'},{'expression':'sample_id','direction':'ASC'}],limit=2,offset=1)
        s=self.service([p,"SELECT region_id,sample_id,changed_px FROM samples WHERE region_id='levir'",
            "SELECT sample_id,changed_px FROM samples WHERE region_id='levir' ORDER BY changed_px DESC,sample_id ASC LIMIT 2 OFFSET 1"])
        r=self.run_query(s,'LEVIR按变化像素从大到小取第2到3张，只返回样本ID、变化像素')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['sqlColumns'],['sample_id','changed_px'])
        self.assertEqual(r['sqlRows'],[{'sample_id':'B','changed_px':0.0},{'sample_id':'C','changed_px':None}])
        self.assertEqual(r['llmCalls'],3)
        self.assertTrue(any(a['status']=='rejected' for a in r['attempts']))

    def test_existing_nullable_metric_is_not_a_missing_field_and_repair_gets_schema_evidence(self):
        p=contract('比较各区域空洞比例均值',regions=[],columns=['region_id','hole_ratio_avg'],calculations=['AVG(hole_ratio)'])
        missing=json.dumps({'sql':None,'clarification':'缺少必要字段hole_ratio'},ensure_ascii=False)
        s=self.service([p,missing,'SELECT region_id,AVG(hole_ratio) AS hole_ratio_avg FROM samples GROUP BY region_id'])
        r=self.run_query(s,'各区域空洞面积比例均值，返回region_id和hole_ratio_avg。')
        self.assertEqual(r['status'],'completed')
        self.assertIn('samples.hole_ratio',self.prompts[-1])
        self.assertIn('NULL',self.prompts[-1])
        self.assertIn('已存在',self.prompts[-1])

    def test_known_metric_with_missing_user_threshold_still_asks_user(self):
        p=contract('LEVIR变化像素筛选',columns=['sample_id'])
        s=self.service([p,json.dumps({'sql':None,'clarification':'请提供changed_px阈值'},ensure_ascii=False)])
        r=self.run_query(s,'LEVIR变化像素筛选，返回样本ID。')
        self.assertEqual(r['status'],'needs_clarification')
        self.assertEqual(r['llmCalls'],2)
        self.assertEqual(r['sqlRows'],[])

    def test_wrong_offset_is_not_reported_as_completed(self):
        p=contract(columns=['sample_id'],order=[{'expression':'sample_id','direction':'ASC'}],limit=1,offset=1)
        s=self.service([p,"SELECT sample_id FROM samples WHERE region_id='levir' ORDER BY sample_id LIMIT 1",
                          "SELECT sample_id FROM samples WHERE region_id='levir' ORDER BY sample_id LIMIT 1"])
        r=self.run_query(s,'LEVIR按样本ID排序的第2张')
        self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

    def test_cross_scope_comparison_clarifies_before_any_model_call(self):
        s=self.service(['SELECT COUNT(*) n FROM samples']*2)
        r=self.run_query(s,'比较LEVIR和WHU的样本数量',scope='whu')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],0)
        self.assertEqual(self.prompts,[])
        self.assertIn('WHU',r['answer'].upper());self.assertIn('LEVIR',r['answer'].upper())

    def test_ordinal_binds_displayed_id_even_after_scalar_followup(self):
        root=contract('LEVIR按ID取前两张',columns=['sample_id','changed_px'],order=[{'expression':'sample_id','direction':'ASC'}],limit=2)
        lookup=contract('查询绑定样本标记',columns=['sample_id','has_holes','has_fragments'])
        s=self.service([root,"SELECT sample_id,changed_px FROM samples WHERE region_id='levir' ORDER BY sample_id LIMIT 2",
            contract('总数',regions=[],columns=['n']),'SELECT COUNT(*) AS n FROM samples',
            lookup,"SELECT sample_id,has_holes,has_fragments FROM samples WHERE sample_id='B'"])
        self.run_query(s,'LEVIR按样本ID取前两张，显示样本ID和变化像素')
        self.run_query(s,'全部样本总数是多少')
        r=self.run_query(s,'刚才第二张有没有空洞和碎片？')
        self.assertEqual(r['status'],'completed')
        self.assertEqual(r['sqlRows'],[{'sample_id':'B','has_holes':0,'has_fragments':0}])
        self.assertEqual(r['resolvedReference']['sampleId'],'B')

    def test_wrong_ordinal_target_cannot_return_another_sample(self):
        root=contract(order=[{'expression':'sample_id','direction':'ASC'}],limit=2)
        s=self.service([root,"SELECT sample_id FROM samples WHERE region_id='levir' ORDER BY sample_id LIMIT 2",
            contract(),"SELECT sample_id FROM samples WHERE sample_id='A'","SELECT sample_id FROM samples WHERE sample_id='A'"])
        self.run_query(s,'LEVIR按ID前两张')
        r=self.run_query(s,'刚才第二张的样本ID')
        self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

    def test_missing_or_out_of_range_ordinal_does_not_guess(self):
        s=self.service([])
        r=self.run_query(s,'刚才第二张有没有空洞？')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],0)

    def test_source_change_invalidates_sample_references(self):
        p=contract(order=[{'expression':'sample_id','direction':'ASC'}],limit=2)
        s=self.service([p,"SELECT sample_id FROM samples WHERE region_id='levir' ORDER BY sample_id LIMIT 2"])
        self.run_query(s,'LEVIR按ID前两张')
        data=json.loads(self.path.read_text(encoding='utf-8'));data['regions'][0]['items'].append({'id':'E'})
        self.path.write_text(json.dumps(data),encoding='utf-8')
        r=self.run_query(s,'刚才第二张是什么')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],0)

    def test_first_list_survives_later_successful_sample_query(self):
        s=self.service([contract('原始条件对象>=3'),"SELECT sample_id FROM samples WHERE region_id='levir' AND object_count>=3",
            contract('另一张'),"SELECT sample_id FROM samples WHERE sample_id='B'",
            contract('原始条件改对象>=2'),"SELECT sample_id FROM samples WHERE region_id='levir' AND object_count>=2"])
        self.run_query(s,'最初独有条件：LEVIR对象至少3个')
        self.run_query(s,'查样本B')
        r=self.run_query(s,'回到最开始的列表，改为至少2个对象')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['sqlRows'],[{'sample_id':'A'},{'sample_id':'C'}])
        self.assertIn('最初独有条件',self.prompts[4])

    def test_sql_stage_cannot_drift_back_to_requesting_average(self):
        p=contract(columns=['n'],calculations=['AVG(changed_px)'])
        s=self.service([p,'{"sql":null,"clarification":"请提供平均值"}',
            "SELECT COUNT(*) n FROM samples WHERE changed_px>(SELECT AVG(changed_px) FROM samples)"])
        r=self.run_query(s,'变化像素高于平均值的样本数')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['llmCalls'],3)

    def test_unavailable_field_clarification_is_specific_and_does_not_run_sql(self):
        p=contract(action='clarify',missing={'kind':'unavailable_field','field':'captured_at','question':'没有拍摄日期字段，无法统计最近7天。'})
        s=self.service([p]);r=self.run_query(s,'最近7天的LEVIR影像数')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],1)

    def test_computed_value_is_not_accepted_as_missing_user_data(self):
        bad=contract(action='clarify',missing={'kind':'computed_value','field':'AVG(changed_px)','question':'请提供平均值'})
        s=self.service([bad,contract(columns=['n'],calculations=['AVG(changed_px)']),
            "SELECT COUNT(*) n FROM samples WHERE changed_px>(SELECT AVG(changed_px) FROM samples)"])
        r=self.run_query(s,'高于平均变化像素的样本数')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['llmCalls'],3)

    def test_existing_column_cannot_be_reported_as_unavailable(self):
        bad=contract(action='clarify',missing={'kind':'unavailable_field','field':'changed_px','question':'请提供变化像素'})
        s=self.service([bad,contract(columns=['n']),'SELECT COUNT(*) n FROM samples'])
        r=self.run_query(s,'样本数量')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['llmCalls'],3)

    def test_known_region_cannot_trigger_region_clarification(self):
        bad=contract(action='clarify',missing={'kind':'user_choice','field':'region','question':'请提供区域'})
        s=self.service([bad,contract(columns=['n']),"SELECT COUNT(*) n FROM samples WHERE region_id='levir'"])
        r=self.run_query(s,'LEVIR样本数量')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['llmCalls'],3)

    def test_scoped_calculation_objects_are_preserved_not_rejected(self):
        p=contract(columns=['n'],calculations=[{'expression':'COUNT(*)','scope':'levir'}])
        s=self.service([p,"SELECT COUNT(*) n FROM samples WHERE region_id='levir'"])
        r=self.run_query(s,'LEVIR样本数')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['sqlRows'],[{'n':3}])

    def test_computed_output_expression_can_match_an_aliased_sql_column(self):
        s=self.service([contract(columns=['SUM(object_count)']),"SELECT SUM(object_count) AS total FROM samples"])
        r=self.run_query(s,'所有对象数量之和')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['sqlRows'],[{'total':8}])

    def test_bound_metric_lookup_may_omit_sample_id(self):
        s=self.service([contract(order=[{'expression':'sample_id','direction':'ASC'}],limit=2),
            "SELECT sample_id FROM samples WHERE region_id='levir' ORDER BY sample_id LIMIT 2",
            contract(columns=['changed_px']),"SELECT changed_px FROM samples WHERE sample_id='B'"])
        self.run_query(s,'LEVIR按ID前两张')
        r=self.run_query(s,'刚才第二张的变化像素，只返回变化像素')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['sqlRows'],[{'changed_px':0.0}])

    def test_bound_id_label_cannot_spoof_another_source_row(self):
        s=self.service([contract(order=[{'expression':'sample_id','direction':'ASC'}],limit=2),
            "SELECT sample_id FROM samples WHERE region_id='levir' ORDER BY sample_id LIMIT 2",
            contract(columns=['sample_id','changed_px']),
            "SELECT 'B' AS sample_id,changed_px FROM samples WHERE sample_id='A'",
            "SELECT 'B' AS sample_id,changed_px FROM samples WHERE sample_id='A'"])
        self.run_query(s,'LEVIR按ID前两张')
        r=self.run_query(s,'刚才第二张的ID和变化像素')
        self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

    def test_unrequested_limit_cannot_silently_drop_rows(self):
        s=self.service([contract(),'SELECT sample_id FROM samples LIMIT 1','SELECT sample_id FROM samples LIMIT 1'])
        r=self.run_query(s,'所有样本ID')
        self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

    def test_group_limit_requires_actual_partitioned_rank_bound(self):
        p=contract(columns=['region_id','sample_id'],per_group=True,limit=2)
        s=self.service([p,'SELECT region_id,sample_id FROM samples','SELECT region_id,sample_id FROM samples'])
        r=self.run_query(s,'每个区域取前两张样本')
        self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

    def test_global_paging_cannot_also_filter_an_inner_window_rank(self):
        p=contract(order=[{'expression':'sample_id','direction':'ASC'}],limit=1,offset=1)
        wrong='WITH x AS (SELECT sample_id,ROW_NUMBER() OVER(ORDER BY sample_id) rn FROM samples) SELECT sample_id FROM x WHERE rn>=2 ORDER BY sample_id LIMIT 1 OFFSET 1'
        s=self.service([p,wrong,wrong]);r=self.run_query(s,'按ID排序的第二张')
        self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

    def test_self_join_sort_cannot_substitute_other_table_same_named_column(self):
        p=contract(order=[{'expression':'a.changed_px','direction':'DESC'},{'expression':'a.sample_id','direction':'ASC'}])
        wrong="SELECT a.sample_id FROM samples a JOIN samples b ON b.sample_id=CASE WHEN a.sample_id='A' THEN 'B' ELSE 'A' END WHERE a.region_id='levir' ORDER BY b.changed_px DESC,a.sample_id ASC"
        s=self.service([p,wrong,wrong]);r=self.run_query(s,'按a表变化像素降序，ID升序')
        self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

    def test_inner_paging_cannot_double_slice_global_page(self):
        p=contract(order=[{'expression':'sample_id','direction':'ASC'}],limit=1,offset=1)
        for sql in [
            "WITH x AS (SELECT sample_id FROM samples ORDER BY sample_id LIMIT 2 OFFSET 1) SELECT sample_id FROM x ORDER BY sample_id LIMIT 1 OFFSET 1",
            "WITH x AS (SELECT sample_id,ROW_NUMBER() OVER(ORDER BY sample_id) rn FROM samples), y AS (SELECT sample_id,rn AS position FROM x) SELECT sample_id FROM y WHERE position>=2 ORDER BY sample_id LIMIT 1 OFFSET 1"]:
            with self.subTest(sql=sql):
                s=self.service([p,sql,sql]);r=self.run_query(s,'按ID排序的第二张')
                self.assertEqual(r['status'],'failed')

    def test_group_rank_cannot_partition_another_source(self):
        p=contract(columns=['region_id','sample_id'],order=[{'expression':'region_id','direction':'ASC'},{'expression':'sample_id','direction':'ASC'}],per_group=True,limit=2)
        sql="WITH ranked AS (SELECT a.region_id,a.sample_id,ROW_NUMBER() OVER(PARTITION BY b.region_id ORDER BY a.sample_id) rn FROM samples a JOIN samples b ON b.sample_id=CASE WHEN a.sample_id='C' THEN 'D' ELSE 'A' END WHERE a.region_id='levir') SELECT region_id,sample_id FROM ranked WHERE rn<=2 ORDER BY region_id,sample_id"
        s=self.service([p,sql,sql]);self.assertEqual(self.run_query(s,'每个区域按ID取前两张')['status'],'failed')

    def test_join_source_identity_survives_table_case_and_cte_aliases(self):
        p=contract(order=[{'expression':'a.changed_px','direction':'DESC'},{'expression':'a.sample_id','direction':'ASC'}])
        for prefix,joined in [('', 'SAMPLES'),('WITH copied AS (SELECT sample_id,changed_px FROM samples) ', 'copied')]:
            sql=prefix+"SELECT a.sample_id FROM samples a JOIN "+joined+" b ON b.sample_id=CASE WHEN a.sample_id='A' THEN 'B' ELSE 'A' END WHERE a.region_id='levir' ORDER BY b.changed_px DESC,a.sample_id ASC"
            with self.subTest(joined=joined):
                s=self.service([p,sql,sql]);self.assertEqual(self.run_query(s,'按a表变化像素降序，ID升序')['status'],'failed')

    def test_sort_literal_case_is_significant(self):
        p=contract(order=[{'expression':"CASE WHEN sample_id='A' THEN 0 ELSE 1 END",'direction':'ASC'},{'expression':'sample_id','direction':'DESC'}])
        sql="SELECT sample_id FROM samples ORDER BY CASE WHEN sample_id='a' THEN 0 ELSE 1 END ASC,sample_id DESC"
        s=self.service([p,sql,sql]);self.assertEqual(self.run_query(s,'先A再按ID降序')['status'],'failed')

    def test_column_expression_with_alias_is_compatible(self):
        s=self.service([contract(columns=['SUM(object_count) AS total']), 'SELECT SUM(object_count) AS total FROM samples'])
        self.assertEqual(self.run_query(s,'对象总数')['status'],'completed')

    def test_sql_stage_keeps_real_region_mapping_and_business_rules(self):
        s=self.service([contract(columns=['n']),"SELECT COUNT(*) n FROM samples WHERE region_id='levir' AND reviewed=0"])
        self.run_query(s,'LEVIR未复核的数量')
        self.assertIn('未复核表示reviewed=0',self.prompts[1])
        self.assertIn('区域：',self.prompts[1])

    def test_bound_reference_keeps_list_definition_when_comparing_its_mean(self):
        p=contract('LEVIR对象至少3个的独有条件',columns=['sample_id'])
        s=self.service([p,"SELECT sample_id FROM samples WHERE region_id='levir' AND object_count>=3",
            contract(columns=['delta']),"SELECT changed_px-(SELECT AVG(changed_px) FROM samples WHERE region_id='levir' AND object_count>=3) AS delta FROM samples WHERE sample_id='A'"])
        self.run_query(s,'LEVIR对象至少3个的独有条件')
        r=self.run_query(s,'刚才第一张的变化像素与该列表平均值差多少？只返回差值。')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['sqlRows'],[{'delta':0.0}])
        self.assertIn('独有条件',self.prompts[2]);self.assertIn('referencedList',self.prompts[2])

    def test_valid_partitioned_top_two_is_accepted(self):
        p=contract(columns=['region_id','sample_id'],order=[{'expression':'region_id','direction':'ASC'},{'expression':'sample_id','direction':'ASC'}],limit=2,per_group=True)
        sql='WITH x AS (SELECT region_id,sample_id,ROW_NUMBER() OVER(PARTITION BY region_id ORDER BY sample_id) rn FROM samples) SELECT region_id,sample_id FROM x WHERE rn<=2 ORDER BY region_id,sample_id'
        s=self.service([p,sql]);r=self.run_query(s,'每个区域按ID取前两张')
        self.assertEqual(r['status'],'completed');self.assertEqual([x['sample_id'] for x in r['sqlRows']],['A','B','D'])

    def test_joined_region_equivalence_does_not_reject_valid_order(self):
        p=contract(columns=['region_name','total'],order=[{'expression':'region_id','direction':'ASC'}])
        sql='SELECT r.region_name,SUM(s.object_count) total FROM samples s JOIN regions r ON s.region_id=r.region_id GROUP BY r.region_id,r.region_name ORDER BY r.region_id'
        s=self.service([p,sql]);self.assertEqual(self.run_query(s,'各区域名称及对象总数，按区域ID排序')['status'],'completed')

    def test_count_expression_contract_is_supported(self):
        s=self.service([contract(columns=['COUNT(*) AS n']),'SELECT COUNT(*) AS n FROM samples'])
        self.assertEqual(self.run_query(s,'样本总数')['status'],'completed')

    def test_missing_schema_clarification_is_not_forced_into_unfiltered_sql(self):
        s=self.service([contract(columns=['n']),json.dumps({'sql':None,'clarification':'数据库没有拍摄日期字段，无法按2026年10月筛选。'},ensure_ascii=False)])
        r=self.run_query(s,'2026年10月拍摄的LEVIR样本数')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],0)
        self.assertEqual(r['clarificationSource'],'unavailable_field_guard')
        self.assertEqual(r['sqlRows'],[]);self.assertEqual(r['sqlQuery'],'')

    def test_complete_query_with_contradictory_action_label_is_normalized(self):
        s=self.service([contract(action='clarify',columns=['n'],missing=None),'SELECT COUNT(*) n FROM samples'])
        r=self.run_query(s,'全部样本数量')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['llmCalls'],2)

    def test_single_global_window_page_with_final_order_is_supported(self):
        p=contract(columns=['sample_id','changed_px'],order=[{'expression':'changed_px','direction':'DESC'},{'expression':'sample_id','direction':'ASC'}],limit=2)
        sql="WITH filtered AS (SELECT sample_id,changed_px FROM samples WHERE region_id='levir'), ranked AS (SELECT sample_id,changed_px,ROW_NUMBER() OVER(ORDER BY changed_px DESC,sample_id) rn FROM filtered) SELECT sample_id,changed_px FROM ranked WHERE rn<=2 ORDER BY changed_px DESC,sample_id"
        s=self.service([p,sql]);r=self.run_query(s,'LEVIR按变化像素降序ID升序前两张，只返回ID、变化像素')
        self.assertEqual(r['status'],'completed');self.assertEqual([x['sample_id'] for x in r['sqlRows']],['A','B'])

    def test_single_window_rank_interval_checks_bounds_and_window_order(self):
        p=contract(order=[{'expression':'sample_id','direction':'ASC'}],limit=2,offset=1)
        base="WITH x AS (SELECT sample_id,ROW_NUMBER() OVER(ORDER BY sample_id {direction}) rn FROM samples WHERE region_id='levir') SELECT sample_id FROM x WHERE rn BETWEEN {low} AND {high} ORDER BY sample_id"
        for direction,low,high,expected in [('ASC',2,3,'completed'),('DESC',2,3,'failed'),('ASC',1,2,'failed')]:
            with self.subTest(direction=direction,low=low):
                sql=base.format(direction=direction,low=low,high=high);s=self.service([p,sql,sql]);r=self.run_query(s,'LEVIR按ID升序第2到3张')
                self.assertEqual(r['status'],expected)
                if expected=='completed':self.assertEqual([x['sample_id'] for x in r['sqlRows']],['B','C'])

    def test_extra_rank_predicates_cannot_shrink_the_requested_page(self):
        p=contract(order=[{'expression':'sample_id','direction':'ASC'}],limit=2,offset=1)
        for predicate in ['rn BETWEEN 2 AND 3 AND rn=3','rn>=2 AND rn<=3 AND rn=3']:
            sql='WITH x AS (SELECT sample_id,ROW_NUMBER() OVER(ORDER BY sample_id) rn FROM samples) SELECT sample_id FROM x WHERE '+predicate+' ORDER BY sample_id'
            with self.subTest(predicate=predicate):
                s=self.service([p,sql,sql]);self.assertEqual(self.run_query(s,'按ID第2到3条')['status'],'failed')

    def test_extra_group_rank_lower_bound_is_rejected(self):
        p=contract(columns=['region_id','sample_id'],order=[{'expression':'region_id','direction':'ASC'}],limit=2,per_group=True)
        sql='WITH x AS (SELECT region_id,sample_id,ROW_NUMBER() OVER(PARTITION BY region_id ORDER BY sample_id) rn FROM samples) SELECT region_id,sample_id FROM x WHERE rn<=2 AND rn>1 ORDER BY region_id'
        s=self.service([p,sql,sql]);self.assertEqual(self.run_query(s,'每个区域按ID取前两条')['status'],'failed')


if __name__=='__main__':unittest.main()
