"""Regression for ordinary terse follow-ups, distinct hole metrics and SQL repair."""
import json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_sql import ModelSQL
from conversation_assistant import ConversationAssistant
from query_dialogue import frame_from_sql,validate_measures,validate_measure_order,DialogueError,context_request,metric_change
from query_contract import resolve_reference

AMBIG='levir区域空洞比例是多少？'
PCT="100.0*SUM(has_holes)/COUNT(*)"
REQUIRED=[dict(operation='flag_percentage',field='has_holes',risk=None)]
def contract(q,regions=None,columns=None,order=None,limit=None):
    return json.dumps(dict(action='query',question=q,regions=regions or [],columns=columns or ['pct'],order=order or [],limit=limit,offset=0,per_group=False,calculations=[],missing=None),ensure_ascii=False)

class HumanWalkthroughTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.prompts=[]
        def item(sid,flag,ratio):return dict(id=sid,maskQuality=dict(hole_ratio=ratio),reviewReasonCodes=['large_hole_ratio'] if flag else [])
        self.source=self.root/'data.json'
        self.source.write_text(json.dumps(dict(regions=[dict(id='levir',short='LEVIR',items=[item('L1',True,.3),item('L2',False,0),item('L3',False,None)]),dict(id='whu',short='WHU',items=[item('W1',True,.6),item('W2',False,0)])])),encoding='utf-8')
    def service(self,outputs):
        seq=iter(outputs)
        def generate(prompt,**kw):self.prompts.append(prompt);return next(seq)
        return ConversationAssistant(ModelSQL(self.source,self.root/'sql.sqlite',generate=generate,model_id='fixture',contract_mode=True),self.root/'chat.sqlite')
    def ask(self,s,q):return s.run(dict(question=q,sessionKey='walk'))
    def test_ambiguous_region_hole_ratio_is_clarified_without_sample_sql(self):
        r=self.ask(self.service([]),AMBIG)
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['clarificationSource'],'metric_definition');self.assertFalse(r['usedSQL']);self.assertEqual(r['llmCalls'],0)
    def test_short_metric_choice_keeps_levir_and_computes_sample_share(self):
        s=self.service([contract('LEVIR有空洞标记的样本占比是多少？',['levir']),f"SELECT {PCT} pct FROM samples WHERE region_id='levir'"])
        self.ask(s,AMBIG);r=self.ask(s,'第一种')
        self.assertEqual(r['status'],'completed',r);self.assertAlmostEqual(r['sqlRows'][0]['pct'],100/3)
        self.assertEqual(r['questionRoute'],'sql');self.assertIn('levir',r['metricResolution']['resolvedQuestion'])
    def test_area_mean_choice_uses_avg_not_sample_flag_count(self):
        s=self.service([contract('LEVIR平均空洞比例是多少？',['levir'],['mean']),"SELECT AVG(hole_ratio) mean FROM samples WHERE region_id='levir'"])
        self.ask(s,AMBIG);r=self.ask(s,'第二种')
        self.assertEqual(r['status'],'completed',r);self.assertAlmostEqual(r['sqlRows'][0]['mean'],.15)
    def test_metric_repair_cannot_replace_aggregate_with_aliased_detail(self):
        q='LEVIR有空洞标记的样本占比是多少？'
        s=self.service([contract(q,['levir'],['hole_ratio']),"SELECT hole_ratio FROM samples WHERE region_id='levir'","SELECT hole_ratio AS pct FROM samples WHERE region_id='levir'"])
        r=self.ask(s,q);self.assertEqual(r['status'],'failed');self.assertFalse(r['sqlRows'])
    def test_compare_other_regions_and_terse_sort_keep_the_same_measure(self):
        first='LEVIR有空洞标记的样本占比是多少？';resolved='对比各区域有空洞标记的样本占比。'
        base=f'SELECT region_id,{PCT} pct FROM samples GROUP BY region_id'
        rewrite=json.dumps(dict(mode='edit',question=resolved,clarification=None),ensure_ascii=False)
        s=self.service([contract(first,['levir']),f"SELECT {PCT} pct FROM samples WHERE region_id='levir'",rewrite,contract(resolved,columns=['region_id','pct']),base,rewrite,contract(resolved+'按占比降序',columns=['region_id','pct'],order=[dict(expression='pct',direction='DESC')]),base+' ORDER BY pct DESC'])
        self.ask(s,first);r=self.ask(s,'那对比其他区域呢？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(len(r['sqlRows']),2)
        r=self.ask(s,'按这个比例从高到低排一下。')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'][0]['region_id'],'whu')
        self.assertIn('有空洞标记的样本百分比',r['contextResolution']['retainedMeasures'])
    def test_percentage_denominator_cannot_be_filtered_or_only_known_area_rows(self):
        invalid=["SELECT 100.0*SUM(has_holes)/COUNT(*) pct FROM samples WHERE has_holes=1",
            'SELECT 100.0*SUM(has_holes)/COUNT(hole_ratio) pct FROM samples',
            'SELECT 100*SUM(has_holes)/COUNT(*) pct FROM samples',
            'SELECT ROUND((COUNT(CASE WHEN has_holes=1 THEN 1 END)/COUNT(*))*100.0,2) pct FROM samples',
            'SELECT ROUND(CAST(COUNT(CASE WHEN has_holes=1 THEN 1 END)/COUNT(*) AS REAL)*100,2) pct FROM samples',
            'SELECT 100.0*AVG(hole_ratio) pct FROM samples',
            'SELECT 100.0*SUM(CASE WHEN has_holes=1 THEN 999 ELSE 0 END)/COUNT(*) pct FROM samples']
        for sql in invalid:
            with self.subTest(sql=sql),self.assertRaises(DialogueError):validate_measures(sql,REQUIRED)
    def test_supported_percentage_shapes_have_same_measure_identity(self):
        for expr in [PCT,'100.0*AVG(has_holes)','100.0*SUM(CASE WHEN has_holes=1 THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0)',
            'CAST(SUM(has_holes) AS REAL)*100/COUNT(*)','CAST(SUM(CASE WHEN has_holes=1 THEN 1 ELSE 0 END) AS REAL)*100.0/COUNT(sample_id)',
            'ROUND((CAST(COUNT(CASE WHEN has_holes=1 THEN 1 END) AS REAL)/COUNT(sample_id))*100,2)']:
            with self.subTest(expr=expr):
                sql=f'SELECT {expr} pct FROM samples';validate_measures(sql,REQUIRED)
                self.assertEqual(frame_from_sql(sql,'有空洞标记的样本占比')['measures'],REQUIRED)

    def test_aggregate_proof_cannot_discard_nontransparent_wrapper_conditions(self):
        inside=f'SELECT {PCT} pct FROM samples'
        for tail in ['WHERE pct>10','LIMIT 1','ORDER BY pct','GROUP BY pct']:
            with self.subTest(tail=tail),self.assertRaises(DialogueError):
                validate_measures(f'SELECT pct FROM ({inside}) {tail}',REQUIRED)
        with self.assertRaises(DialogueError):
            validate_measures(f'SELECT pct FROM ({inside} WHERE has_holes=1)',REQUIRED)
    def test_second_sample_possessive_binds_existing_list_without_requiring_just_now(self):
        history=dict(last_list=dict(list_id='x',ordered_ids=['L1','L2','L3']))
        ref,error=resolve_reference('第二张的变化像素和对象数量呢？',history)
        self.assertIsNone(error);self.assertEqual(ref['sampleId'],'L2')
    def test_a_new_rank_interval_is_not_an_implicit_sample_reference(self):
        ref,error=resolve_reference('WHU变化像素第2到4张的样本ID是什么？',dict(last_list=dict(list_id='x',ordered_ids=['L1','L2'])))
        self.assertIsNone(ref);self.assertIsNone(error)
    def test_comparison_cannot_report_success_while_still_only_reading_levir(self):
        first='LEVIR有空洞标记的样本占比是多少？';resolved='LEVIR与其他区域的有空洞标记的样本占比是多少？'
        rewrite=json.dumps(dict(mode='edit',question=resolved,clarification=None),ensure_ascii=False)
        wrong=f"SELECT region_id,{PCT} pct FROM samples WHERE region_id='levir' GROUP BY region_id"
        right=f'SELECT region_id,{PCT} pct FROM samples GROUP BY region_id'
        s=self.service([contract(first,['levir']),f"SELECT {PCT} pct FROM samples WHERE region_id='levir'",rewrite,contract(resolved,['levir'],['region_id','pct']),wrong,right])
        self.ask(s,first);r=self.ask(s,'那对比其他区域呢？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(len(r['sqlRows']),2)
        self.assertTrue(any(a.get('error','').startswith('本轮对比各区域') for a in r['attempts']))
    def test_metric_ranking_cannot_silently_sort_by_total_sample_count(self):
        wrong=f'SELECT region_id,{PCT} pct FROM samples GROUP BY region_id ORDER BY COUNT(*) DESC'
        with self.assertRaises(DialogueError):validate_measure_order(wrong,REQUIRED,'DESC')
        validate_measure_order(wrong.replace('COUNT(*) DESC','pct DESC'),REQUIRED,'DESC')
    def test_comparison_may_redundantly_list_all_visible_region_ids(self):
        first='LEVIR有空洞标记的样本占比是多少？';resolved='各区域有空洞标记的样本占比是多少？'
        rewrite=json.dumps(dict(mode='edit',question=resolved,clarification=None),ensure_ascii=False)
        sql=f"SELECT region_id,{PCT} pct FROM samples WHERE region_id IN ('whu','levir') GROUP BY region_id"
        s=self.service([contract(first,['levir']),f"SELECT {PCT} pct FROM samples WHERE region_id='levir'",rewrite,contract(resolved,columns=['region_id','pct']),sql])
        self.ask(s,first);r=self.ask(s,'那对比其他区域呢？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(len(r['sqlRows']),2)
    def test_region_pronoun_is_bound_to_the_last_real_single_region(self):
        first='各区域有空洞标记的样本占比最高的是哪个区域？'
        base=f'SELECT region_id,{PCT} pct FROM samples GROUP BY region_id ORDER BY pct DESC LIMIT 1'
        q='这个区域里，挑空洞比例最大的3张未复核样本。'
        c=contract(first,columns=['region_id','pct'],order=[dict(expression='pct',direction='DESC')],limit=1)
        detail=contract('WHU区域空洞比例最大的3张未复核样本',['whu'],['sample_id','hole_ratio'],[dict(expression='hole_ratio',direction='DESC')],3)
        detail_object=json.loads(detail);detail_object['per_group']=True;detail=json.dumps(detail_object,ensure_ascii=False)
        wrong='SELECT sample_id,hole_ratio FROM samples WHERE reviewed=0 ORDER BY hole_ratio DESC LIMIT 3'
        right=wrong.replace('WHERE reviewed=0',"WHERE reviewed=0 AND region_id='whu'")
        s=self.service([c,base,detail,wrong,right]);self.ask(s,first);r=self.ask(s,q)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['resolvedRegionReference']['regionId'],'whu')
        self.assertEqual([row['sample_id'] for row in r['sqlRows']],['W1','W2'])
        self.assertFalse(r['queryContract']['per_group']);self.assertTrue(r['population']['supported'])
        self.assertTrue(any(a.get('error','').startswith('主查询区域约束') for a in r['attempts']))
    def test_region_pronoun_does_not_guess_from_a_multi_region_report(self):
        first='各区域有空洞标记的样本占比是多少？'
        s=self.service([contract(first,columns=['region_id','pct']),f'SELECT region_id,{PCT} pct FROM samples GROUP BY region_id'])
        self.ask(s,first);r=self.ask(s,'这个区域里空洞比例最大的样本是哪张？')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['clarificationSource'],'region_reference_guard')
    def test_numeric_average_among_flagged_samples_is_not_a_flag_percentage(self):
        frame=frame_from_sql('SELECT AVG(hole_ratio) mean FROM samples WHERE has_holes=1','有空洞标记的样本平均空洞比例')
        self.assertEqual(frame['measures'],[dict(operation='avg',field='hole_ratio',risk=None)])
    def test_risk_switch_after_thanks_preserves_total_count_region_selection(self):
        first='样本最多的区域的高风险样本数量和占比是多少？'
        high="SELECT region_id,SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END) n,100.0*SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)/COUNT(*) pct FROM samples GROUP BY region_id ORDER BY COUNT(*) DESC LIMIT 1"
        c=contract(first,columns=['region_id','n','pct'],order=[dict(expression='COUNT(*)',direction='DESC')],limit=1)
        rewrite=json.dumps(dict(mode='edit',question='各区域低风险数量和占比是多少？',clarification=None),ensure_ascii=False)
        low=high.replace("'high'","'low'")
        s=self.service([c,high,'不客气。',rewrite,c.replace('高风险','低风险'),low])
        self.ask(s,first);self.ask(s,'谢谢！');r=self.ask(s,'那低风险呢？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(len(r['sqlRows']),1)
        self.assertEqual(r['sqlRows'][0]['region_id'],'levir');self.assertIn('样本数量最多',r['contextResolution']['resolvedQuestion'])
        self.assertEqual(r['contextResolution']['retainedMeasures'],['低风险样本数量','低风险样本百分比'])
    def test_missing_capture_year_is_rejected_before_model_can_drop_it(self):
        r=self.ask(self.service([]),'WHU里2025年拍摄、带空洞标记的样本占比是多少？')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['clarificationSource'],'unavailable_field_guard')
        self.assertEqual(r['llmCalls'],0);self.assertFalse(r['usedSQL'])
    def test_cancelling_the_missing_year_allows_the_now_complete_question(self):
        q='不按年份了，先查WHU有空洞标记的样本占比。'
        s=self.service([contract(q,['whu']),f"SELECT {PCT} pct FROM samples WHERE region_id='whu'"])
        self.ask(s,'WHU里2025年拍摄、带空洞标记的样本占比是多少？');r=self.ask(s,q)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[dict(pct=50.0)])
    def test_average_of_flagged_samples_keeps_area_mean_not_flag_sample_percentage(self):
        q='LEVIR有空洞标记的样本，其平均空洞比例是多少？'
        s=self.service([contract(q,['levir'],['mean']),"SELECT AVG(hole_ratio) mean FROM samples WHERE region_id='levir' AND has_holes=1"])
        r=self.ask(s,q);self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[dict(mean=.3)])
    def test_comparing_flagged_sample_means_does_not_invent_a_flag_percentage_goal(self):
        q='LEVIR有空洞标记的样本，其平均空洞比例是多少？';resolved='各区域有空洞标记的样本平均空洞比例对比。'
        rewrite=json.dumps(dict(mode='edit',question=resolved,clarification=None),ensure_ascii=False)
        s=self.service([contract(q,['levir'],['mean']),"SELECT AVG(hole_ratio) mean FROM samples WHERE region_id='levir' AND has_holes=1",rewrite,contract(resolved,columns=['region_id','mean']),'SELECT region_id,AVG(hole_ratio) mean FROM samples WHERE has_holes=1 GROUP BY region_id'])
        self.ask(s,q);r=self.ask(s,'那对比其他区域呢？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual({row['region_id']:row['mean'] for row in r['sqlRows']},{'levir':.3,'whu':.6})
    def test_named_region_does_not_require_a_previous_pronoun_binding(self):
        q='LEVIR中高于该区域平均变化像素的样本有多少张？'
        sql="SELECT COUNT(*) n FROM samples WHERE region_id='levir' AND changed_px>(SELECT AVG(changed_px) FROM samples WHERE region_id='levir')"
        r=self.ask(self.service([contract(q,['levir'],['n']),sql]),q)
        self.assertEqual(r['status'],'completed',r);self.assertNotIn('resolvedRegionReference',r)

    def risk_report(self):
        q='各区域高风险样本的数量和占比分别是多少？按占比降序排列。'
        expr="SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)"
        sql=f'SELECT region_id,{expr} n,100.0*{expr}/COUNT(*) pct FROM samples GROUP BY region_id ORDER BY pct DESC'
        return q,contract(q,columns=['region_id','n','pct'],order=[dict(expression='pct',direction='DESC')]),sql

    def test_exact_reported_switch_clarifies_new_metric_and_keeps_levir_after_choice(self):
        first,c,sql=self.risk_report()
        final=contract('LEVIR有空洞标记的样本占比',['levir'])
        s=self.service([c,sql,final,f"SELECT {PCT} pct FROM samples WHERE region_id='levir'"])
        self.ask(s,first);r=self.ask(s,'那levir的空洞区域比例呢')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['clarificationSource'],'metric_definition')
        self.assertFalse(r['usedSQL']);self.assertNotIn('contextResolution',r)
        r=self.ask(s,'第一种')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(len(r['sqlRows']),1)
        self.assertAlmostEqual(r['sqlRows'][0]['pct'],100/3)
        self.assertEqual(r['queryContract']['regions'],['levir']);self.assertEqual(r['metricUpdate']['families'],['holes'])
        self.assertNotIn('review_level',r['sqlQuery'])

    def test_new_metric_clarification_replaces_an_unrelated_pending_question(self):
        first,c,sql=self.risk_report()
        s=self.service([c,sql,contract('LEVIR有空洞标记的样本占比',['levir']),f"SELECT {PCT} pct FROM samples WHERE region_id='levir'"])
        self.ask(s,first);self.ask(s,'WHU里2025年拍摄的样本有多少？')
        self.ask(s,'那levir的空洞区域比例呢');r=self.ask(s,'第一种')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['queryContract']['regions'],['levir'])
        self.assertNotIn('2025',r['metricResolution']['resolvedQuestion'])

    def test_explicit_area_mean_replaces_risk_instead_of_appending_it(self):
        first,c,sql=self.risk_report();q='那levir的平均空洞面积比例呢？'
        s=self.service([c,sql,contract(q,['levir'],['mean']),"SELECT AVG(hole_ratio) mean FROM samples WHERE region_id='levir'"])
        self.ask(s,first);r=self.ask(s,q)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[dict(mean=.15)])
        self.assertNotIn('contextResolution',r);self.assertNotIn('高风险',r['queryContract']['question'])

    def test_explicit_numeric_mean_replaces_risk_goal(self):
        first,c,sql=self.risk_report();q='那WHU平均变化像素呢？'
        s=self.service([c,sql,contract(q,['whu'],['mean']),"SELECT AVG(changed_px) mean FROM samples WHERE region_id='whu'"])
        self.ask(s,first);r=self.ask(s,q)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['metricUpdate']['families'],['changed_px'])
        self.assertNotIn('高风险',r['queryContract']['question'])

    def test_percentage_refinement_deduplicates_count_and_percentage_goals(self):
        first,c,sql=self.risk_report()
        request=context_request('那levir的高风险占比呢？',dict(sql=sql,question=first))
        self.assertEqual(request['required'],[dict(operation='risk_percentage',field='review_level',risk='high')])

    def test_current_named_region_survives_a_rewrite_that_returns_old_all_region_report(self):
        first,c,sql=self.risk_report();q='那levir的高风险占比呢？'
        rewrite=json.dumps(dict(mode='edit',question=first,clarification=None),ensure_ascii=False)
        pct_sql="SELECT region_id,100.0*SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)/COUNT(*) pct FROM samples GROUP BY region_id ORDER BY pct DESC"
        right=pct_sql.replace('FROM samples GROUP BY',"FROM samples WHERE region_id='levir' GROUP BY")
        c2=contract(first,columns=['region_id','pct'],order=[dict(expression='pct',direction='DESC')])
        s=self.service([c,sql,rewrite,c2,pct_sql,right]);self.ask(s,first);r=self.ask(s,q)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[dict(region_id='levir',pct=0.0)])
        self.assertEqual(r['queryContract']['regions'],['levir']);self.assertEqual(r['contextResolution']['retainedMeasures'],['高风险样本百分比'])
        self.assertIn('levir',r['contextResolution']['resolvedQuestion'])

    def test_marker_used_as_a_filter_does_not_replace_risk_measure(self):
        first,c,sql=self.risk_report();history=dict(sql=sql,question=first)
        for q in ['只看有空洞标记的样本','只看空洞比例高于0.1的样本']:
            with self.subTest(q=q):
                self.assertIsNone(metric_change(q,history))
                self.assertEqual(context_request(q,history)['required'],frame_from_sql(sql,first)['measures'])

    def test_explicit_all_region_metric_change_does_not_inherit_old_single_region(self):
        first='LEVIR平均空洞比例是多少？';q,c,sql=self.risk_report()
        s=self.service([contract(first,['levir'],['mean']),"SELECT AVG(hole_ratio) mean FROM samples WHERE region_id='levir'",c,sql])
        self.ask(s,first);r=self.ask(s,q)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(len(r['sqlRows']),2)
        self.assertEqual(r['queryContract']['regions'],[]);self.assertNotIn('沿用区域',r['queryContract']['question'])

    def test_risk_level_change_keeps_current_named_region_even_if_rewrite_omits_it(self):
        q='LEVIR高风险占比是多少？'
        high="SELECT 100.0*SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)/COUNT(*) pct FROM samples WHERE region_id='levir'"
        rewrite=json.dumps(dict(mode='edit',question='低风险占比是多少？',clarification=None),ensure_ascii=False)
        low=high.replace("'high'","'low'")
        s=self.service([contract(q,['levir']),high,rewrite,contract('低风险占比是多少？'),low])
        self.ask(s,q);r=self.ask(s,'那低风险呢？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['queryContract']['regions'],['levir'])
        self.assertIn('region_id = \'levir\'',r['sqlQuery'])

    def test_internal_scope_note_does_not_invent_an_outer_sort_requirement(self):
        q='LEVIR高风险占比是多少？'
        high="SELECT 100.0*SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)/COUNT(*) pct FROM samples WHERE region_id='levir'"
        rewrite=json.dumps(dict(mode='edit',question='LEVIR低风险占比；本轮候选区域沿用已执行查询：levir；排序或只取最高不会新增区域筛选。',clarification=None),ensure_ascii=False)
        generated=contract('LEVIR低风险占比',columns=['pct'],order=[dict(expression='pct',direction='DESC')])
        s=self.service([contract(q,['levir']),high,rewrite,generated,high.replace("'high'","'low'")])
        self.ask(s,q);r=self.ask(s,'那低风险呢？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['queryContract']['order'],[])

if __name__=='__main__':unittest.main()
