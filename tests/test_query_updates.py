"""Rank deltas must preserve scope, filters and the executed aggregate body."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_sql import ModelSQL
from conversation_assistant import ConversationAssistant
from query_updates import report_task,parse_update,validate_update_sql,contract_for_update,literal_rank,eligible,UpdateError,validate_rank_contract

BASE='SELECT region_id,AVG(hole_ratio) AS mean FROM samples GROUP BY region_id'
REGIONS=[dict(id=r,short=r.upper()) for r in ('uav','clcd','levir')]

def delta(n=2,direction='DESC',end='same',target=None):
    return dict(action='verify_rank' if target else 'update_query',changes=dict(rank=dict(start=n,end=n if end=='same' else end,direction=direction)),target_region=target,clarification=None)

def contract(q,limit=None,offset=0):
    return json.dumps(dict(action='query',question=q,regions=[],columns=['region_id','mean'],order=[dict(expression='mean',direction='DESC')],limit=limit,offset=offset,per_group=False,calculations=[],missing=None),ensure_ascii=False)

class QueryUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.prompts=[];self.options=[]
        self.source=self.root/'source.json'
        self.source.write_text(json.dumps(dict(regions=[dict(id=r,short=r.upper(),items=[dict(id=r+'1',maskQuality=dict(hole_ratio=value))]) for r,value in [('uav',.3),('clcd',.2),('levir',.1)]])),encoding='utf-8')
        self.task=report_task(dict(sql=BASE+' ORDER BY mean DESC LIMIT 1',queryContract=dict(question='各区域空洞面积比例均值排名',regions=[])))
    def service(self,outputs):
        seq=iter(outputs)
        def generate(p,**kw):self.prompts.append(p);self.options.append(kw);return next(seq)
        return ConversationAssistant(ModelSQL(self.source,self.root/'sql.sqlite',generate=generate,model_id='fixture',contract_mode=True),self.root/'chat.sqlite')
    def ask(self,s,q,**extra):return s.run(dict(question=q,sessionKey='rank')|extra)
    def beginning(self,outputs):
        q='各区域的空洞面积比例均值按均值降序'
        s=self.service([contract(q),BASE+' ORDER BY mean DESC']+outputs)
        self.assertEqual(self.ask(s,q)['status'],'completed');return s
    def test_second_rank_bypasses_chat_classifier_and_full_replanning(self):
        s=self.beginning([json.dumps(delta()),BASE+' ORDER BY mean DESC,region_id ASC LIMIT 1 OFFSET 1'])
        r=self.ask(s,'排名第二的呢');self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['sqlRows'],[dict(region_id='clcd',mean=.2)]);self.assertEqual(r['llmCalls'],2)
        self.assertEqual(r['queryState']['offset'],1);self.assertIn('第2名是CLCD',r['answer'])
        self.assertEqual([x['stage'] for x in r['attempts']],['state_update','sql'])
    def test_wrong_rank_delta_is_repaired_before_sql_generation(self):
        s=self.beginning([json.dumps(delta(2,end=4)),json.dumps(delta()),BASE+' ORDER BY mean DESC,region_id ASC LIMIT 1 OFFSET 1'])
        r=self.ask(s,'那第二高的呢？');self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['queryContract']['limit'],1);self.assertTrue(any(x['status']=='rejected' for x in r['attempts']))
    def test_wrong_sql_page_cannot_execute_even_when_model_plan_is_valid(self):
        s=self.beginning([json.dumps(delta()),BASE+' ORDER BY mean DESC,region_id ASC LIMIT 3 OFFSET 1',BASE+' ORDER BY mean DESC,region_id ASC LIMIT 1 OFFSET 1'])
        r=self.ask(s,'第二高的呢');self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlExecutionCount'],1)
    def test_highest_after_second_restores_first_and_keeps_global_candidates(self):
        s=self.beginning([json.dumps(delta()),BASE+' ORDER BY mean DESC,region_id ASC LIMIT 1 OFFSET 1',json.dumps(delta(1)),BASE+' ORDER BY mean DESC,region_id ASC LIMIT 1'])
        self.ask(s,'第二高的呢');r=self.ask(s,'最高的是哪个')
        self.assertEqual(r['sqlRows'],[dict(region_id='uav',mean=.3)]);self.assertEqual(r['queryState']['regions'],[])
    def test_rank_assertion_is_verified_over_all_candidates(self):
        s=self.beginning([json.dumps(delta(target='uav')),BASE+' ORDER BY mean DESC,region_id ASC'])
        r=self.ask(s,'uav不是第二高的吗');self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['rankVerification']['actualRank'],1);self.assertFalse(r['rankVerification']['matchesClaim']);self.assertEqual(len(r['sqlRows']),3)
        self.assertEqual(r['queryContract']['regions'],[])
    def test_chat_between_rank_edits_preserves_task(self):
        s=self.beginning(['不客气。',json.dumps(delta()),BASE+' ORDER BY mean DESC,region_id ASC LIMIT 1 OFFSET 1'])
        self.assertEqual(self.ask(s,'谢谢')['status'],'chat_completed');self.assertEqual(self.ask(s,'排名第二的呢')['sqlRows'][0]['region_id'],'clcd')
    def test_scope_and_filter_body_cannot_be_changed(self):
        task=report_task(dict(sql=BASE.replace('GROUP BY',"WHERE region_id IN ('uav','clcd') AND reviewed=0 GROUP BY")+' ORDER BY mean DESC',queryContract=dict(regions=['uav','clcd'],question='区域均值')))
        c=contract_for_update('第二高',task,delta())
        with self.assertRaises(UpdateError):validate_update_sql(BASE+' ORDER BY mean DESC,region_id ASC LIMIT 1 OFFSET 1',task,c,None)
    def test_verification_target_cannot_become_a_filter(self):
        c=contract_for_update('uav不是第二高',self.task,delta(target='uav'))
        with self.assertRaises(UpdateError):validate_update_sql(BASE.replace('GROUP BY',"WHERE region_id='uav' GROUP BY")+' ORDER BY mean DESC,region_id ASC',self.task,c,None)
    def test_numeric_mean_cannot_be_replaced_by_marker_percentage(self):
        c=contract_for_update('第二高',self.task,delta())
        with self.assertRaises(UpdateError):validate_update_sql(BASE.replace('AVG(hole_ratio)','100.0*AVG(has_holes)')+' ORDER BY mean DESC,region_id ASC LIMIT 1 OFFSET 1',self.task,c,None)
    def test_plain_ordinal_inherits_ascending_direction(self):
        task=copy.deepcopy(self.task);task['rank']['direction']='ASC'
        parsed=parse_update(json.dumps(delta(direction='ASC')),'排名第二的呢',task,REGIONS)
        self.assertEqual(parsed['changes']['rank']['direction'],'ASC')
    def test_next_rank_uses_current_page_end(self):
        task=copy.deepcopy(self.task);task['rank']=dict(start=2,end=4,direction='DESC')
        self.assertEqual(parse_update(json.dumps(delta(5)),'再往后看一位',task,REGIONS)['changes']['rank']['start'],5)
        with self.assertRaises(UpdateError):parse_update(json.dumps(delta(3)),'再往后看一位',task,REGIONS)
    def test_metric_change_and_explanation_still_use_their_own_paths(self):
        self.assertFalse(eligible('哪个区域的对象数量均值最高',self.task,REGIONS));self.assertFalse(eligible('解释排名第二是什么意思',self.task,REGIONS))
        self.assertFalse(eligible('第二种',self.task,REGIONS));self.assertFalse(eligible('第二高的呢',None,REGIONS))
    def test_bad_or_forged_delta_fields_are_not_accepted(self):
        bad=delta();bad['changes']['regions']=['uav']
        with self.assertRaises(UpdateError):parse_update(json.dumps(bad),'第二高',self.task,REGIONS)
        bad=delta(target='clcd')
        with self.assertRaises(UpdateError):parse_update(json.dumps(bad),'uav不是第二高',self.task,REGIONS)
    def test_invalid_rank_range_is_rejected(self):
        with self.assertRaises(UpdateError):literal_rank('第4到2名')
    def test_invalid_range_is_a_user_clarification_without_model_or_sql(self):
        s=self.beginning([]);r=self.ask(s,'第4到2名')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['clarificationSource'],'invalid_rank');self.assertEqual(r['llmCalls'],0);self.assertFalse(r['usedSQL'])
    def test_adapter_is_only_requested_for_the_state_update_stage(self):
        s=self.beginning([json.dumps(delta()),BASE+' ORDER BY mean DESC,region_id ASC LIMIT 1 OFFSET 1']);s.sql.query_update_adapter=True
        self.assertEqual(self.ask(s,'第二高')['status'],'completed')
        self.assertTrue(self.options[-2]['use_query_adapter']);self.assertNotIn('use_query_adapter',self.options[-1]);self.assertNotIn('use_query_adapter',self.options[0])
    def test_denominator_count_does_not_become_a_new_user_goal_after_rank_edit(self):
        q='各区域高风险样本的数量和占比分别是多少？按占比降序'
        expr="SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)"
        body=f'SELECT region_id,{expr} high_count,COUNT(*) total,100.0*{expr}/COUNT(*) pct FROM samples GROUP BY region_id'
        c=json.loads(contract(q));c['columns']=['region_id','high_count','total','pct'];c['order']=[dict(expression='pct',direction='DESC')]
        s=self.service([json.dumps(c),body+' ORDER BY pct DESC',json.dumps(delta()),body+' ORDER BY pct DESC,region_id ASC LIMIT 1 OFFSET 1'])
        self.assertEqual(self.ask(s,q)['status'],'completed');r=self.ask(s,'第二高的呢')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(len(r['queryState']['measures']),2)
        self.assertEqual({m['value']['operation'] for m in r['queryState']['measures']},{'risk_count','risk_percentage'})
    def test_rank_intervals_share_the_same_parser_at_routing_and_validation(self):
        for q in ['第2到4名都给我看看','第2至4名','第2—4位']:
            self.assertTrue(eligible(q,self.task,REGIONS))
            self.assertEqual(parse_update(json.dumps(delta(2,end=4)),q,self.task,REGIONS)['changes']['rank'],dict(start=2,end=4,direction='DESC'))
    def test_initial_contract_is_checked_against_explicit_single_rank(self):
        bad=json.loads(contract('哪个区域的空洞均值排名第二',3,1))
        with self.assertRaises(UpdateError):validate_rank_contract(bad['question'],bad)
        bad.update(limit=1);validate_rank_contract(bad['question'],bad)
    def test_no_cross_session_query_state(self):
        s=self.beginning(['CLARIFY']);r=self.ask(s,'排名第二的呢',sessionKey='different')
        self.assertEqual(r['status'],'needs_clarification');self.assertFalse(r['usedSQL'])
    def test_secondary_tie_break_does_not_override_primary_sort_direction(self):
        cases=[('各区域均值按mean_ratio降序，均值相同按region_id升序。','DESC'),
               ('各区域均值按mean_ratio升序，均值相同按region_id降序。','ASC')]
        for question,direction in cases:
            with self.subTest(question=question):
                self.assertEqual(literal_rank(question)['direction'],direction)
                plan=dict(action='query',columns=['region_id','mean_ratio'],per_group=False,order=[dict(expression='mean_ratio',direction=direction),dict(expression='region_id',direction='ASC' if direction=='DESC' else 'DESC')],limit=None,offset=0)
                validate_rank_contract(question,plan)
                plan['order'][0]['direction']='ASC' if direction=='DESC' else 'DESC'
                with self.assertRaises(UpdateError):validate_rank_contract(question,plan)
    def test_semantic_rank_direction_survives_opposite_tie_break(self):
        cases=[('变化比例最低的区域，均值相同按region_id降序。','ASC'),
               ('变化像素最少的前2个区域，并列时region_id降序。','ASC'),
               ('变化比例最高的区域，均值相同按region_id升序。','DESC'),
               ('均值最高的区域，按region_id升序打破并列。','DESC')]
        for question,direction in cases:
            with self.subTest(question=question):
                self.assertEqual(literal_rank(question)['direction'],direction)
    def test_explicit_restart_uses_full_planner_instead_of_rank_delta_adapter(self):
        for question in ['重新查询本批全部样本：空洞均值最低的区域。','请重新统计，空洞均值最高的区域。','帮我从头查询全部区域的第二名。']:
            with self.subTest(question=question):
                self.assertFalse(eligible(question,self.task,REGIONS))
        self.assertTrue(eligible('不要重新查询，第二低的呢？',self.task,REGIONS))

if __name__=='__main__':unittest.main()
