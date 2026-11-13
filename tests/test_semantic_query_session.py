import sys,unittest,json
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from query_semantic_protocol import task
from semantic_query_session import SemanticQuerySession

class SemanticSessionTests(unittest.TestCase):
    def test_chat_action_delegates_to_real_chat_without_sql(self):
        sql=Mock();chat=Mock(return_value=dict(status='chat_completed',usedSQL=False,answer='你好'))
        s=SemanticQuerySession([dict(id='whu',label='WHU')],lambda _:json.dumps(task(action='chat')),sql,chat=chat)
        r=s.run('你好');self.assertEqual(r['answer'],'你好');chat.assert_called_once();sql.run.assert_not_called()
    def test_adapter_only_plans_sql_model_still_executes_and_failure_does_not_commit(self):
        t=task(regions=['whu'],measure='sample_count');sql=Mock()
        sql.run.return_value=dict(status='failed',sqlQuery='',sqlRows=[])
        s=SemanticQuerySession([dict(id='whu',label='WHU')],lambda _:json.dumps(t),sql)
        r=s.run('几张',view_region='whu');self.assertIsNone(s.previous_task)
        self.assertIn('WHU',sql.run.call_args.args[0]['question']);self.assertFalse(r['usedSQL'])
    def test_clarification_survives_chat_and_success_commits_actual_selection(self):
        outputs=iter([task(action='clarify',clarification='region'),task(action='chat'),task(regions=['whu'],measure='changed_px')])
        sql=Mock();sql.run.return_value=dict(status='completed',sqlQuery='SELECT sample_id,changed_px FROM samples',sqlRows=[dict(sample_id='whu-a',changed_px=10)])
        s=SemanticQuerySession([dict(id='whu',label='WHU')],lambda _:json.dumps(next(outputs)),sql)
        s.run('这个区域有几张');s.run('你好');self.assertEqual(s.pending_question,'这个区域有几张')
        self.assertEqual(sql.run.call_count,0);s.run('WHU');self.assertIsNone(s.pending_question);self.assertEqual(s.previous_selection,['whu-a'])
    def test_invalid_or_scope_escaping_output_never_calls_sql(self):
        sql=Mock();s=SemanticQuerySession([dict(id='whu',label='WHU'),dict(id='levir',label='LEVIR')],lambda _:json.dumps(task(regions=['levir'],measure='sample_count')),sql)
        self.assertEqual(s.run('查LEVIR',scope_region='whu')['status'],'semantic_rejected');sql.run.assert_not_called()
    def test_aggregate_on_previous_selection_keeps_the_original_selected_ids(self):
        outputs=iter([task(regions=['whu'],measure='changed_px'),task(regions=['whu'],measure='sum_changed_px',population='previous_selection')])
        sql=Mock();sql.run.side_effect=[dict(status='completed',sqlQuery='SELECT sample_id FROM samples',sqlRows=[dict(sample_id='whu-a')]),dict(status='completed',sqlQuery='SELECT SUM(changed_px) FROM samples',sqlRows=[dict(total=10)])]
        s=SemanticQuerySession([dict(id='whu',label='WHU')],lambda _:json.dumps(next(outputs)),sql)
        s.run('选样本');s.run('这些样本求和');self.assertEqual(s.previous_selection,['whu-a'])
if __name__=='__main__':unittest.main()
