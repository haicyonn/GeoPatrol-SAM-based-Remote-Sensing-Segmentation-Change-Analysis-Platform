import sys,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from query_semantic_protocol import task,fixture_database,execute_task
from evaluate_semantic_conversations import export_fixture,REGIONS

class DirectSQLTests(unittest.TestCase):
    def setUp(self):
        import semantic_direct_sql as direct
        self.d=direct;self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'data.json';export_fixture(self.path,0)
        self.c=dict(regions=REGIONS,view_region='whu',scope_region=None,previous_task=None,previous_question=None,previous_selection=[])
    def executor(self,outputs):
        from model_sql import ModelSQL
        generated=Mock(side_effect=outputs);base=ModelSQL(self.path,Path(self.temp.name)/'h.db',generate=generated,model_id='test',contract_mode=True)
        base.run=Mock(side_effect=AssertionError('Legacy planner must never run'))
        return self.d.SemanticSQLExecutor(base),base,generated
    def test_global_region_rank_is_not_per_group_and_sql_is_model_written(self):
        t=task(measure='hole_ratio_mean',group_by='region',rank=dict(direction='DESC',start=1,end=1))
        sql='SELECT region_id,AVG(hole_ratio) AS hole_ratio_mean FROM selected_samples GROUP BY region_id ORDER BY hole_ratio_mean DESC,region_id ASC LIMIT 1'
        ex,base,gen=self.executor([sql]);r=ex.run_task(t,self.c)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'][0]['region_id'],'uav');self.assertEqual(gen.call_count,1);base.run.assert_not_called()
        self.assertFalse(r['queryContract']['per_group']);self.assertNotIn('SELECT region_id,AVG(hole_ratio)',gen.call_args.args[0])
    def test_server_binds_region_and_review_filters_without_new_interpretation(self):
        self.c['scope_region']='whu';t=task(regions=['whu'],measure='sample_count',filters=dict(reviewed=0,risk_level=None))
        ex,_,_=self.executor(['SELECT COUNT(*) AS sample_count FROM selected_samples'])
        r=ex.run_task(t,self.c);self.assertEqual(r['sqlRows'],[dict(sample_count=2)],r)
    def test_original_selection_is_separate_from_full_region_mean(self):
        self.c['previous_selection']=['whu-a','whu-b'];t=task(regions=['whu'],measure='above_region_mean_count',population='previous_selection',reference='region_all_mean')
        sql='SELECT COUNT(*) AS above_region_mean_count FROM selected_samples s WHERE s.changed_px>(SELECT AVG(r.changed_px) FROM reference_samples r WHERE r.region_id=s.region_id)'
        ex,_,_=self.executor([sql]);r=ex.run_task(t,self.c);self.assertEqual(r['sqlRows'],[dict(above_region_mean_count=0)],r)
    def test_wrong_metric_unbound_source_extra_filter_and_write_never_execute(self):
        t=task(regions=['whu'],measure='sample_count')
        for sql in ['SELECT SUM(changed_px) AS sample_count FROM selected_samples','SELECT COUNT(*) AS sample_count FROM samples','SELECT COUNT(*) AS sample_count FROM selected_samples WHERE reviewed=1','DELETE FROM selected_samples']:
            ex,_,_=self.executor([sql,sql]);r=ex.run_task(t,self.c);self.assertEqual(r['status'],'failed',r);self.assertFalse(r['usedSQL'])
    def test_uncorrelated_reference_is_rejected(self):
        self.c['previous_selection']=['whu-a','whu-b'];t=task(regions=['whu'],measure='above_region_mean_count',population='previous_selection',reference='region_all_mean')
        sql='SELECT COUNT(*) AS above_region_mean_count FROM selected_samples s WHERE s.changed_px>(SELECT AVG(changed_px) FROM reference_samples)'
        ex,_,_=self.executor([sql,sql]);r=ex.run_task(t,self.c);self.assertEqual(r['status'],'failed',r)
    def test_unsupported_qualify_cannot_filter_the_returned_count(self):
        sql='SELECT COUNT(*) AS sample_count FROM selected_samples QUALIFY sample_count>100'
        ex,_,_=self.executor([sql,sql]);r=ex.run_task(task(measure='sample_count'),self.c)
        self.assertEqual(r['status'],'failed',r);self.assertFalse(r['usedSQL'])
    def test_nulls_first_cannot_turn_unknown_mean_into_highest_region(self):
        raw=json.loads(self.path.read_text())
        for item in raw['regions'][0]['items']:item['maskQuality']['hole_ratio']=None
        self.path.write_text(json.dumps(raw))
        t=task(measure='hole_ratio_mean',group_by='region',rank=dict(direction='DESC',start=1,end=1))
        sql='SELECT region_id,AVG(hole_ratio) AS hole_ratio_mean FROM selected_samples GROUP BY region_id ORDER BY hole_ratio_mean DESC NULLS FIRST,region_id ASC LIMIT 1'
        ex,_,_=self.executor([sql,sql]);r=ex.run_task(t,self.c);self.assertEqual(r['status'],'failed',r)
    def test_redundant_identity_alias_and_commuted_percentage_are_valid(self):
        t=task(measure='high_risk_percent',group_by='region')
        sql="SELECT region_id AS region_id,SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)*100.0/NULLIF(COUNT(sample_id),0) AS high_risk_percent FROM selected_samples GROUP BY region_id ORDER BY region_id ASC"
        ex,_,_=self.executor([sql]);r=ex.run_task(t,self.c);self.assertEqual(r['status'],'completed',r)
    def test_explicit_no_execution_keeps_existing_chat_policy_before_adapter(self):
        from semantic_query_session import DirectSemanticQuerySession
        interpret=Mock(side_effect=AssertionError('No-execution policy cannot be overridden by adapter'))
        sql=Mock();chat=Mock(return_value=dict(status='chat_completed',usedSQL=False,answer='仅展示示例'))
        session=DirectSemanticQuerySession(REGIONS,interpret,sql,chat=chat);session.pending_question='WHU的空洞比例是多少'
        r=session.run('只写SQL，统计WHU图片数量，不要执行')
        self.assertEqual(r['status'],'chat_completed');interpret.assert_not_called();sql.run.assert_not_called()
        self.assertEqual(session.pending_question,'WHU的空洞比例是多少')
    def test_following_chat_can_see_the_actual_direct_query_result(self):
        from model_sql import ModelSQL
        from conversation_assistant import ConversationAssistant
        from semantic_query_session import DirectSemanticQuerySession
        generate=Mock(side_effect=['SELECT COUNT(*) AS sample_count FROM selected_samples','这里的数量是样本组数。'])
        sql=ModelSQL(self.path,Path(self.temp.name)/'chat-sql.db',generate=generate,model_id='test',contract_mode=True)
        chat=ConversationAssistant(sql,Path(self.temp.name)/'chat.db')
        outputs=iter([task(regions=['whu'],measure='sample_count'),task(action='chat')])
        session=DirectSemanticQuerySession(REGIONS,lambda _:json.dumps(next(outputs)),sql,chat=chat.run)
        session.run('WHU有多少样本',session_key='remember')
        session.run('解释刚才结果的含义',session_key='remember')
        history=generate.call_args.kwargs['history_messages']
        self.assertTrue(any('sample_count' in r['content'] and '3' in r['content'] for r in history),history)
    def test_sql_model_does_not_receive_already_bound_scope_or_null_filters(self):
        t=task(regions=['whu'],measure='sample_count')
        ex,_,gen=self.executor(['SELECT COUNT(*) AS sample_count FROM selected_samples']);ex.run_task(t,self.c)
        prompt=gen.call_args.args[0]
        self.assertNotIn('"regions"',prompt);self.assertNotIn('"filters"',prompt);self.assertNotIn('whu',prompt)
    def test_real_cast_percentage_is_equivalent_but_integer_division_is_not(self):
        t=task(regions=['whu'],measure='high_risk_percent')
        sql="SELECT CAST(SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END) AS REAL)*100/COUNT(*) AS high_risk_percent FROM selected_samples"
        ex,_,_=self.executor([sql]);r=ex.run_task(t,self.c);self.assertEqual(r['status'],'completed',r)
        self.assertAlmostEqual(r['sqlRows'][0]['high_risk_percent'],200/3)
        wrong=sql.replace('CAST(SUM(CASE WHEN review_level=\'high\' THEN 1 ELSE 0 END) AS REAL)','SUM(CASE WHEN review_level=\'high\' THEN 1 ELSE 0 END)')
        ex,_,_=self.executor([wrong,wrong]);r=ex.run_task(t,self.c);self.assertEqual(r['status'],'failed',r)
    def test_query_memory_rejects_stale_or_wrong_scope_and_releases_lock(self):
        from model_sql import ModelSQL,QueryError
        from conversation_assistant import ConversationAssistant
        sql=ModelSQL(self.path,Path(self.temp.name)/'reject-sql.db',generate=Mock(),model_id='test')
        chat=ConversationAssistant(sql,Path(self.temp.name)/'reject-chat.db');payload=dict(question='样本数',sessionKey='reject',scopeRegion='whu')
        for changed in [dict(scopeRegion='levir'),dict(status='failed'),{}]:
            result=dict(status='completed',usedSQL=True,sqlQuery='SELECT COUNT(*) FROM samples',scopeRegion='whu',dataVersion='stale',**{})
            result.update(changed)
            with self.assertRaises(QueryError):chat.record_query_result(payload,result)
            self.assertTrue(chat._gate.acquire(blocking=False));chat._gate.release()
        self.assertEqual(chat._history(sql._owner(payload),'whu'),[])
    def test_invalid_task_retries_schema_once_before_any_sql(self):
        from semantic_query_session import DirectSemanticQuerySession
        invalid=task(regions=['whu'],measure='sum_changed_px',rank=dict(direction='DESC',start=1,end=2))
        valid=task(regions=['whu'],measure='changed_px',rank=dict(direction='DESC',start=1,end=2))
        interpret=Mock(side_effect=[json.dumps(invalid),json.dumps(valid)])
        ex,base,gen=self.executor(['SELECT sample_id,changed_px FROM selected_samples ORDER BY changed_px DESC,sample_id ASC LIMIT 2'])
        session=DirectSemanticQuerySession(REGIONS,interpret,base,schema_repair=True)
        result=session.run('WHU按变化像素取前2名，列出ID和变化像素')
        self.assertEqual(result['status'],'completed',result);self.assertEqual(result['semanticLLMCalls'],2);self.assertEqual(gen.call_count,1)
        self.assertIn('scalar ranking',interpret.call_args.args[0]);self.assertEqual(len(result['interpretationAttempts']),2)
    def test_two_invalid_tasks_fail_closed_and_valid_task_is_never_rejudged(self):
        from semantic_query_session import DirectSemanticQuerySession
        interpret=Mock(return_value='not-json');sql=Mock()
        session=DirectSemanticQuerySession(REGIONS,interpret,sql,schema_repair=True);r=session.run('查样本')
        self.assertEqual(r['status'],'semantic_rejected');self.assertEqual(interpret.call_count,2);sql.run.assert_not_called()
        interpret=Mock(return_value=json.dumps(task(regions=['whu'],measure='sample_count')))
        _,base,_=self.executor(['SELECT COUNT(*) AS sample_count FROM selected_samples'])
        session=DirectSemanticQuerySession(REGIONS,interpret,base);r=session.run('查询WHU样本数')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(interpret.call_count,1)
if __name__=='__main__':unittest.main()
