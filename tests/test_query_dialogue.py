"""A scope correction must not delete a previously requested measure."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_sql import ModelSQL
from query_dialogue import frame_from_sql,context_request,parse_resolution

FIRST='各个区域高风险量分别是多少？'
FOLLOW='我想要的是样本最多的区域'
RESOLVED='样本最多的区域的高风险样本数量是多少？'
COUNT="SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)"
FIRST_SQL=f'SELECT region_id,{COUNT} AS n FROM samples GROUP BY region_id ORDER BY region_id'
FINAL_SQL=f'SELECT region_id,{COUNT} AS n FROM samples GROUP BY region_id ORDER BY COUNT(*) DESC LIMIT 1'

def contract(q,top=False,columns=None):
    return json.dumps(dict(action='query',question=q,regions=[],columns=columns or ['region_id','n'],
        order=[dict(expression='COUNT(*)' if top else 'region_id',direction='DESC' if top else 'ASC')],
        limit=1 if top else None,offset=0,per_group=False,calculations=[],missing=None),ensure_ascii=False)

def resolution(q=RESOLVED):return json.dumps(dict(mode='edit',question=q,clarification=None),ensure_ascii=False)

class DialogueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'data.json'
        self.source.write_text(json.dumps({'regions':[
            {'id':'whu','short':'WHU','items':[dict(id='W'+str(i),reviewLevel=l,changedPx=(i+1)*10) for i,l in enumerate(['high','low','none','unknown'])]},
            {'id':'levir','short':'LEVIR','items':[dict(id='L'+str(i),reviewLevel=l,changedPx=(i+1)*100) for i,l in enumerate(['high','high','low'])]}]}),encoding='utf-8')
        self.prompts=[]
    def service(self,outputs):
        seq=iter(outputs)
        def generate(p,**kw):self.prompts.append(p);return next(seq)
        return ModelSQL(self.source,self.root/'history.sqlite',generate=generate,model_id='fixture',contract_mode=True)
    def ask(self,s,q,session='s'):return s.run(dict(question=q,sessionKey=session))
    def test_correction_keeps_risk_count_but_changes_region_selection(self):
        s=self.service([contract(FIRST),FIRST_SQL,resolution(),contract(RESOLVED,True),FINAL_SQL])
        self.assertEqual(self.ask(s,FIRST)['status'],'completed')
        r=self.ask(s,FOLLOW);self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['sqlRows'],[dict(region_id='whu',n=1)])
        self.assertEqual(r['contextResolution']['retainedMeasures'],['高风险样本数量'])
        self.assertEqual(r['llmCalls'],3)
    def test_rewrite_changing_measure_is_rejected_and_repaired(self):
        s=self.service([contract(FIRST),FIRST_SQL,resolution('样本最多的区域的低风险样本数量是多少？'),resolution(),contract(RESOLVED,True),FINAL_SQL])
        self.ask(s,FIRST);r=self.ask(s,FOLLOW)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['attempts'][0]['status'],'rejected')
        self.assertEqual(r['attempts'][0]['stage'],'context')
    def test_program_binds_omitted_measure_and_discloses_model_rewrite(self):
        s=self.service([contract(FIRST),FIRST_SQL,resolution('哪个区域的样本数量最多？'),contract(RESOLVED,True),FINAL_SQL])
        self.ask(s,FIRST);r=self.ask(s,FOLLOW)
        self.assertEqual(r['status'],'completed',r)
        self.assertTrue(r['contextResolution']['programBoundMeasures'])
        self.assertNotIn('高风险',r['contextResolution']['modelResolvedQuestion'])
        self.assertIn('高风险样本数量',r['contextResolution']['resolvedQuestion'])
    def test_explicit_selection_is_not_lost_when_model_rewrites_all_regions(self):
        q='不是按高风险量选，按全部样本数选最多的区域，其他不变'
        s=self.service([contract(FIRST),FIRST_SQL,resolution('各区域样本总数分别是多少？'),contract(RESOLVED,True),FINAL_SQL])
        self.ask(s,FIRST);r=self.ask(s,q)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[dict(region_id='whu',n=1)])
    def test_unrequested_inferred_sort_is_not_a_business_constraint(self):
        c=json.loads(contract(FIRST));c['order']=[dict(expression='sample_count',direction='DESC')]
        sql=FIRST_SQL.replace(' ORDER BY region_id','')
        r=self.ask(self.service([json.dumps(c),sql]),FIRST)
        self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['queryContract']['order'],[])
    def test_contract_cannot_rewrite_bound_task_again(self):
        s=self.service([contract(FIRST),FIRST_SQL,resolution(),contract('哪个区域的样本数量最多？',True),FINAL_SQL])
        self.ask(s,FIRST);r=self.ask(s,FOLLOW)
        self.assertEqual(r['status'],'completed',r)
        self.assertIn('高风险',r['queryContract']['question'])
        self.assertIn('question_from_bound_task',r['queryContract']['normalizations'])
    def test_sql_cannot_turn_risk_count_into_total_count(self):
        s=self.service([contract(FIRST),FIRST_SQL,resolution(),contract(RESOLVED,True),
            'SELECT region_id,COUNT(*) n FROM samples GROUP BY region_id ORDER BY COUNT(*) DESC LIMIT 1',FINAL_SQL])
        self.ask(s,FIRST);r=self.ask(s,FOLLOW)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'][0]['n'],1)
        self.assertTrue(any(a['stage']=='sql' and a['status']=='rejected' for a in r['attempts']))
    def test_sql_cannot_rank_by_high_risk_population(self):
        wrong="SELECT region_id,COUNT(*) n FROM samples WHERE review_level='high' GROUP BY region_id ORDER BY COUNT(*) DESC LIMIT 1"
        s=self.service([contract(FIRST),FIRST_SQL,resolution(),contract(RESOLVED,True),wrong,FINAL_SQL])
        self.ask(s,FIRST);r=self.ask(s,FOLLOW)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'][0]['region_id'],'whu')
    def test_percentage_is_preserved_as_percentage(self):
        psql=f'SELECT region_id,100.0*{COUNT}/COUNT(*) n FROM samples GROUP BY region_id ORDER BY region_id'
        q='各个区域高风险占比是多少？';resolved='样本最多的区域的高风险占比是多少？'
        final=psql.replace('ORDER BY region_id','ORDER BY COUNT(*) DESC LIMIT 1')
        s=self.service([contract(q),psql,resolution(resolved),contract(resolved,True),final])
        self.ask(s,q);r=self.ask(s,FOLLOW)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'][0]['n'],25.0)
    def test_numeric_aggregate_is_retained_not_replaced_by_region_count(self):
        q='各区域平均变化像素是多少？';resolved='样本最多的区域的平均变化像素是多少？'
        sql='SELECT region_id,AVG(changed_px) n FROM samples GROUP BY region_id ORDER BY region_id'
        final=sql.replace('ORDER BY region_id','ORDER BY COUNT(*) DESC LIMIT 1')
        s=self.service([contract(q),sql,resolution(resolved),contract(resolved,True),final])
        self.ask(s,q);r=self.ask(s,FOLLOW)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'][0]['n'],25.0)
    def test_explicit_risk_change_replaces_risk_not_count_operation(self):
        q='改成低风险，其余不变';resolved='各个区域低风险样本数量分别是多少？'
        s=self.service([contract(FIRST),FIRST_SQL,resolution(resolved),contract(resolved),FIRST_SQL.replace("'high'","'low'")])
        self.ask(s,FIRST);r=self.ask(s,q)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['contextResolution']['retainedMeasures'],['低风险样本数量'])
    def test_explicit_new_question_does_not_inherit_old_risk(self):
        q='另外查询各区域样本总数';sql='SELECT region_id,COUNT(*) n FROM samples GROUP BY region_id ORDER BY region_id'
        s=self.service([contract(FIRST),FIRST_SQL,contract(q),sql]);self.ask(s,FIRST);r=self.ask(s,q)
        self.assertEqual(r['status'],'completed',r);self.assertNotIn('contextResolution',r);self.assertEqual(r['llmCalls'],2)
    def test_task_frames_do_not_cross_sessions(self):
        s=self.service([contract(FIRST),FIRST_SQL,contract('哪个区域的样本数最多',True),
            'SELECT region_id,COUNT(*) n FROM samples GROUP BY region_id ORDER BY COUNT(*) DESC LIMIT 1'])
        self.ask(s,FIRST);r=self.ask(s,FOLLOW,session='other')
        self.assertEqual(r['status'],'completed',r);self.assertNotIn('contextResolution',r)
    def test_measure_frame_handles_where_and_conditional_count_equivalence(self):
        a=frame_from_sql("SELECT region_id,COUNT(*) n FROM samples WHERE review_level='high' GROUP BY region_id",FIRST)
        b=frame_from_sql(FIRST_SQL,FIRST)
        self.assertEqual(a['measures'],b['measures'])
    def test_percentage_memory_does_not_promote_auxiliary_counts_to_goals(self):
        sql=f'SELECT region_id,{COUNT} n,COUNT(*) total,100.0*{COUNT}/COUNT(*) pct FROM samples GROUP BY region_id'
        frame=frame_from_sql(sql,'各区域高风险占比是多少？')
        self.assertEqual(frame['measures'],[dict(operation='risk_percentage',field='review_level',risk='high')])
    def test_explicit_count_and_percentage_both_remain_goals(self):
        sql=f'SELECT region_id,{COUNT} n,COUNT(*) total,100.0*{COUNT}/COUNT(*) pct FROM samples GROUP BY region_id'
        frame=frame_from_sql(sql,'各区域高风险数量和占比是多少？')
        self.assertEqual([m['operation'] for m in frame['measures']],['risk_count','risk_percentage'])
    def test_canonical_region_selection_retains_additional_user_constraints(self):
        q='我想要的是WHU和LEVIR中样本最多的区域，只看最近7天'
        request=context_request(q,dict(sql=FIRST_SQL,question=FIRST))
        resolved=parse_resolution(resolution('哪个区域样本最多？'),request)
        self.assertIn('WHU和LEVIR',resolved['question'])
        self.assertIn('最近7天',resolved['question'])
    def test_sample_list_and_explicit_reset_skip_aggregate_rewriting(self):
        self.assertIsNone(context_request('换成LEVIR',dict(sql='SELECT sample_id FROM samples LIMIT 10',question='前10张')))
        self.assertIsNone(context_request('新问题：我想要的是样本最多的区域',dict(sql=FIRST_SQL,question=FIRST)))

if __name__=='__main__':unittest.main()
