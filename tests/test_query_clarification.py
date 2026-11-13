"""Typed choices bind one missing slot to one owner/scope/data revision."""
import json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_sql import ModelSQL
from conversation_assistant import ConversationAssistant
from query_clarification import semantic_state

def contract(q,columns=('pct',),regions=('levir',),order=(),action='query',missing=None):
    return json.dumps(dict(action=action,question=q,columns=list(columns),regions=list(regions),order=list(order),limit=None,offset=0,per_group=False,calculations=[],missing=missing),ensure_ascii=False)

class ClarificationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.calls=[]
        self.source=self.root/'data.json'
        self.source.write_text(json.dumps(dict(regions=[
            dict(id='levir',short='LEVIR',items=[dict(id='L1',changedPx=10,maskQuality=dict(hole_ratio=.3,small_fragment_area_ratio=.2),reviewReasonCodes=['large_hole_ratio','many_small_fragments'],reviewLevel='high'),dict(id='L2',changedPx=200,maskQuality=dict(hole_ratio=0,small_fragment_area_ratio=0),reviewLevel='low')]),
            dict(id='whu',short='WHU',items=[dict(id='W1',changedPx=30,maskQuality=dict(hole_ratio=.6,small_fragment_area_ratio=.4),reviewReasonCodes=['large_hole_ratio'],reviewLevel='low')])])),encoding='utf-8')
    def service(self,outputs=()):
        sequence=iter(outputs)
        def generate(prompt,**kw):self.calls.append(prompt);return next(sequence)
        sql=ModelSQL(self.source,self.root/'sql.sqlite',generate=generate,model_id='fixture',contract_mode=True)
        return ConversationAssistant(sql,self.root/'chat.sqlite')
    def ask(self,s,q,**kw):return s.run(dict(question=q,sessionKey='s',_authenticatedUser='alice',**kw))
    def select(self,s,pending,index=0,**kw):
        option=pending['options'][index]
        return self.ask(s,option['label'],clarificationId=pending['id'],optionId=option['id'],**kw)

    def test_catalog_returns_structured_choices_before_any_model_or_sample_sql(self):
        r=self.ask(self.service(),'LEVIR空洞区域比例是多少？')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],0);self.assertFalse(r['usedSQL'])
        c=r['clarification'];self.assertEqual(c['slot'],'measure_definition');self.assertEqual(len(c['options']),2)
        self.assertEqual(c['task']['regions'],['levir']);self.assertNotIn('choiceValues',c)

    def test_a_bare_region_and_metric_request_does_not_need_conversation_history(self):
        r=self.ask(self.service(),'那levir的空洞区域比例呢')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['clarification']['slot'],'measure_definition')
        self.assertEqual(r['llmCalls'],0);self.assertFalse(r['usedSQL'])

    def test_explaining_a_catalog_metric_stays_chat(self):
        s=self.service(['CHAT','这是项目的数据域和掩膜形态指标。'])
        r=self.ask(s,'解释LEVIR的空洞比例是什么，不要查询数据')
        self.assertEqual(r['status'],'chat_completed');self.assertFalse(r['usedSQL'])

    def test_selection_is_a_bound_slot_update_and_state_is_grounded_in_executed_sql(self):
        q='LEVIR有空洞标记的样本占比'
        s=self.service([contract(q),"SELECT 100.0*SUM(has_holes)/COUNT(*) pct FROM samples WHERE region_id='levir'"])
        pending=self.ask(s,'LEVIR空洞区域比例是多少？')['clarification'];r=self.select(s,pending)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[dict(pct=50.0)])
        self.assertEqual(r['clarificationResolution']['id'],pending['id']);self.assertNotIn('clarification',r)
        self.assertEqual(r['queryState']['regions'],['levir']);self.assertEqual(r['queryState']['measures'][0]['value']['field'],'has_holes')
        self.assertEqual(self.ask(s,'',action='history')['queryState'],r['queryState'])

    def test_fragment_metric_uses_the_same_protocol_without_hole_specific_reply_logic(self):
        q='LEVIR碎片面积比例均值'
        s=self.service([contract(q,columns=['mean']),"SELECT AVG(small_fragment_area_ratio) mean FROM samples WHERE region_id='levir'"])
        r=self.ask(s,'LEVIR碎片比例是多少？');self.assertEqual(len(r['clarification']['options']),2)
        r=self.select(s,r['clarification'],1);self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['sqlRows'],[dict(mean=.1)]);self.assertEqual(r['queryState']['measures'][0]['value']['field'],'small_fragment_area_ratio')

    def test_chosen_percentage_repairs_integer_division_before_executing(self):
        q='LEVIR有空洞标记的样本占比'
        bad="SELECT ROUND((SUM(has_holes)/COUNT(*))*100.0,2) pct FROM samples WHERE region_id='levir'"
        good="SELECT ROUND(100.0*SUM(has_holes)/NULLIF(COUNT(*),0),2) pct FROM samples WHERE region_id='levir'"
        s=self.service([contract(q),bad,good])
        pending=self.ask(s,'LEVIR空洞比例是多少？')['clarification']
        r=self.select(s,pending)
        self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['sqlRows'],[dict(pct=50.0)])
        self.assertEqual(r['sqlExecutionCount'],1)
        self.assertIn('除法之前',self.calls[-1])
        self.assertIn('100.0*SUM',self.calls[-1])

    def test_confirmed_share_accepts_transparent_projection_and_float_count_division(self):
        q='LEVIR有空洞标记的样本占比'
        columns=['has_holes_count','sample_count','has_holes_percentage']
        sql="SELECT has_holes_count,sample_count,has_holes_percentage FROM (SELECT COUNT(CASE WHEN has_holes=1 THEN 1 ELSE NULL END) has_holes_count,COUNT(sample_id) sample_count,ROUND((CAST(COUNT(CASE WHEN has_holes=1 THEN 1 ELSE NULL END) AS REAL)/COUNT(sample_id))*100,2) has_holes_percentage FROM samples WHERE region_id='levir')"
        s=self.service([contract(q,columns=columns),sql])
        p=self.ask(s,'那levir的空洞区域比例呢')['clarification'];r=self.select(s,p)
        self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['sqlRows'],[dict(has_holes_count=1,sample_count=2,has_holes_percentage=50.0)])
        self.assertEqual(r['queryState']['regions'],['levir']);self.assertEqual(r['queryState']['measures'][0]['value']['field'],'has_holes')

    def test_fragment_marker_percentage_is_not_hole_percentage_or_area_mean(self):
        q='LEVIR有碎片标记的样本占比'
        s=self.service([contract(q),"SELECT 100.0*SUM(has_fragments)/COUNT(*) pct FROM samples WHERE region_id='levir'"])
        r=self.select(s,self.ask(s,'LEVIR碎片比例是多少？')['clarification'])
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[dict(pct=50.0)])

    def test_risk_grade_is_a_typed_choice(self):
        q='LEVIR高风险占比'
        s=self.service([contract(q),"SELECT 100.0*SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)/COUNT(*) pct FROM samples WHERE region_id='levir'"])
        p=self.ask(s,'LEVIR风险占比是多少？')['clarification'];self.assertEqual(len(p['options']),3)
        r=self.select(s,p);self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[dict(pct=50.0)])

    def test_choice_does_not_accept_a_client_supplied_definition(self):
        s=self.service();p=self.ask(s,'LEVIR空洞比例是多少？')['clarification']
        r=self.ask(s,'随便',clarificationId=p['id'],optionId='forged',choiceValue=dict(sql='SELECT 1'))
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['clarificationSource'],'stale_clarification')
        self.assertEqual(r['llmCalls'],0);self.assertFalse(r['usedSQL']);self.assertEqual(r['clarification']['id'],p['id'])

    def test_choice_is_isolated_from_other_sessions_and_scopes(self):
        s=self.service();p=self.ask(s,'LEVIR空洞比例是多少？')['clarification'];o=p['options'][0]
        for extra in [dict(sessionKey='other'),dict(_authenticatedUser='bob'),dict(scopeRegion='whu')]:
            payload=dict(question=o['label'],sessionKey='s',_authenticatedUser='alice',clarificationId=p['id'],optionId=o['id']);payload.update(extra)
            r=s.run(payload);self.assertEqual(r['clarificationSource'],'stale_clarification');self.assertFalse(r['usedSQL'])

    def test_choice_expires_after_snapshot_changes(self):
        s=self.service();p=self.ask(s,'LEVIR空洞比例是多少？')['clarification']
        data=json.loads(self.source.read_text(encoding='utf-8'));data['updated']='new';self.source.write_text(json.dumps(data),encoding='utf-8')
        r=self.select(s,p);self.assertEqual(r['clarificationSource'],'stale_clarification');self.assertEqual(r['llmCalls'],0)

    def test_successful_choice_cannot_be_replayed(self):
        s=self.service([contract('LEVIR有空洞标记样本占比'),"SELECT 100.0*SUM(has_holes)/COUNT(*) pct FROM samples WHERE region_id='levir'"])
        p=self.ask(s,'LEVIR空洞比例是多少？')['clarification'];self.assertEqual(self.select(s,p)['status'],'completed')
        r=self.select(s,p);self.assertEqual(r['clarificationSource'],'stale_clarification');self.assertFalse(r['usedSQL']);self.assertEqual(r['llmCalls'],0)

    def test_a_failed_new_query_still_invalidates_old_choices(self):
        s=self.service(['invalid','invalid']);p=self.ask(s,'LEVIR空洞比例是多少？')['clarification']
        self.assertEqual(self.ask(s,'WHU样本数量是多少？')['status'],'failed')
        r=self.select(s,p);self.assertEqual(r['clarificationSource'],'stale_clarification');self.assertFalse(r['usedSQL'])

    def test_new_question_supersedes_old_choices(self):
        s=self.service();old=self.ask(s,'LEVIR空洞比例是多少？')['clarification'];new=self.ask(s,'WHU碎片比例是多少？')['clarification']
        self.assertNotEqual(old['id'],new['id']);r=self.select(s,old)
        self.assertEqual(r['clarificationSource'],'stale_clarification');self.assertEqual(r['clarification']['id'],new['id'])

    def test_supplement_and_history_preserve_pending_without_executing_sql(self):
        s=self.service(['不客气。']);p=self.ask(s,'LEVIR空洞比例是多少？')['clarification']
        r=self.ask(s,'谢谢');self.assertEqual(r['status'],'chat_completed');self.assertFalse(r['usedSQL'])
        self.assertEqual(r['clarification']['id'],p['id'])
        restored=self.ask(s,'',action='history');self.assertEqual(restored['clarification'],p)

    def test_manual_ordinal_uses_current_choices_and_out_of_range_is_not_guessed(self):
        s=self.service();p=self.ask(s,'LEVIR空洞比例是多少？')['clarification'];r=self.ask(s,'第三种')
        self.assertEqual(r['clarificationSource'],'invalid_clarification_choice');self.assertEqual(r['clarification']['id'],p['id']);self.assertFalse(r['usedSQL'])

    def test_threshold_supplement_keeps_the_original_question(self):
        q='LEVIR中变化像素大于阈值的样本有多少张？'
        missing=dict(kind='user_choice',field='threshold',question='变化像素要大于多少？')
        s=self.service([contract(q,columns=[],action='clarify',missing=missing),contract(q,columns=['n']),"SELECT COUNT(*) n FROM samples WHERE region_id='levir' AND changed_px>100"])
        r=self.ask(s,q);self.assertEqual(r['clarification']['slot'],'threshold');self.assertTrue(r['clarification']['acceptsText'])
        r=self.ask(s,'100');self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[dict(n=1)])

    def test_numeric_threshold_one_is_a_value_not_an_option_index(self):
        q='LEVIR中变化像素大于阈值的样本有多少张？'
        missing=dict(kind='user_choice',field='threshold',question='变化像素要大于多少？')
        s=self.service([contract(q,columns=[],action='clarify',missing=missing),contract(q,columns=['n']),"SELECT COUNT(*) n FROM samples WHERE region_id='levir' AND changed_px>1"])
        self.ask(s,q);r=self.ask(s,'1');self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[dict(n=2)])

    def test_sort_ambiguity_can_choose_count_instead_of_defaulting_to_percentage(self):
        q='各区域高风险数量和占比分别是多少？'
        first="SELECT region_id,SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END) n,100.0*SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)/COUNT(*) pct FROM samples GROUP BY region_id"
        c=contract(q,columns=['region_id','n','pct'],regions=[])
        rewrite=json.dumps(dict(mode='edit',question='各区域高风险数量和占比，按高风险样本数量降序',clarification=None),ensure_ascii=False)
        ranked=contract(q,columns=['region_id','n','pct'],regions=[],order=[dict(expression='n',direction='DESC')])
        s=self.service([c,first,rewrite,ranked,first+' ORDER BY n DESC'])
        self.ask(s,q);pending=self.ask(s,'从高到低排一下')['clarification']
        self.assertEqual(pending['slot'],'sort_metric');self.assertEqual(len(pending['options']),2)
        r=self.select(s,pending,0);self.assertEqual(r['status'],'completed',r);self.assertEqual(len(r['sqlRows']),2)
        self.assertEqual(r['contextResolution']['retainedMeasures'],['高风险样本数量','高风险样本百分比'])

    def test_region_pronoun_offers_actual_candidate_regions(self):
        q='各区域样本数量是多少？'
        s=self.service([contract(q,columns=['region_id','n'],regions=[]),'SELECT region_id,COUNT(*) n FROM samples GROUP BY region_id',contract('WHU样本数量',['n'],['whu']),"SELECT COUNT(*) n FROM samples WHERE region_id='whu'"])
        self.ask(s,q);r=self.ask(s,'这个区域有多少样本？');p=r['clarification']
        self.assertEqual(p['slot'],'region');self.assertEqual([o['label'] for o in p['options']],['LEVIR','WHU'])
        # Selection only binds the region; aggregate rewriting remains model-authored.
        self.assertFalse(r['usedSQL'])

    def test_or_branch_is_not_presented_as_a_mandatory_filter(self):
        history=dict(sql='SELECT COUNT(*) n FROM samples WHERE reviewed=0 OR has_holes=1',question='未复核或有空洞',queryContract=dict(regions=[]))
        state=semantic_state(history,[dict(id='levir',short='LEVIR')]);self.assertEqual(state['filters'],[])

    def test_risk_filter_does_not_replace_a_numeric_mean_goal(self):
        q='LEVIR高风险样本的平均变化像素是多少？'
        s=self.service([contract(q,columns=['mean']),"SELECT AVG(changed_px) mean FROM samples WHERE region_id='levir' AND review_level='high'"])
        r=self.ask(s,q);self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[dict(mean=10.0)])
        state=r['queryState'];self.assertEqual(state['measures'][0]['value']['field'],'changed_px');self.assertIn('高风险',state['filters'])

    def test_sql_format_error_is_repaired_internally_without_asking_user(self):
        q='LEVIR样本有多少张？'
        bad=json.dumps(dict(sql=None,clarification='需要补充ORDER BY和SQL别名才能通过校验。'),ensure_ascii=False)
        s=self.service([contract(q,columns=['n']),bad,"SELECT COUNT(*) n FROM samples WHERE region_id='levir'"])
        r=self.ask(s,q);self.assertEqual(r['status'],'completed',r);self.assertNotIn('clarification',r)
        self.assertTrue(any(a['status']=='rejected' for a in r['attempts']));self.assertEqual(r['sqlRows'],[dict(n=2)])

    def test_repeated_internal_sql_failure_is_not_presented_as_missing_user_input(self):
        q='LEVIR样本有多少张？';bad=json.dumps(dict(sql=None,clarification='JSON格式不正确，需要更多信息。'),ensure_ascii=False)
        r=self.ask(self.service([contract(q,columns=['n']),bad,bad]),q)
        self.assertEqual(r['status'],'failed');self.assertNotIn('clarification',r);self.assertFalse(r['usedSQL'])

    def test_missing_calculated_alias_is_repaired_without_asking_for_a_schema_change(self):
        q='LEVIR样本有多少张？';bad=json.dumps(dict(sql=None,clarification="缺少字段 'sample_count'，请提供该字段。"),ensure_ascii=False)
        s=self.service([contract(q,columns=['sample_count']),bad,"SELECT COUNT(*) sample_count FROM samples WHERE region_id='levir'"])
        r=self.ask(s,q);self.assertEqual(r['status'],'completed',r);self.assertNotIn('clarification',r)
        self.assertEqual(r['sqlRows'],[dict(sample_count=2)])

    def test_repeated_missing_calculated_alias_is_a_technical_failure(self):
        q='LEVIR有空洞标记的样本占比';bad=json.dumps(dict(sql=None,clarification="缺少字段 'has_holes_count' 和 'has_holes_percentage'，请检查表结构。"),ensure_ascii=False)
        s=self.service([contract(q,columns=['has_holes_count','has_holes_percentage']),bad,bad])
        r=self.ask(s,q);self.assertEqual(r['status'],'failed',r);self.assertNotIn('clarification',r)

if __name__=='__main__':unittest.main()
