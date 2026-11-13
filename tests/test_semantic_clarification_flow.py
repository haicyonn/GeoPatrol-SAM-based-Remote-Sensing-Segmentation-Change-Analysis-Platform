"""The experimental bridge must preserve the existing clarification contract."""
import json
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from query_semantic_protocol import task
from semantic_query_session import DirectSemanticQuerySession


class ClarificationFlowTests(unittest.TestCase):
    def session(self, outputs):
        model = Mock(side_effect=[json.dumps(t, ensure_ascii=False) for t in outputs])
        sql = Mock()
        session = DirectSemanticQuerySession(
            [dict(id='whu', label='WHU'), dict(id='levir', label='LEVIR')], model, sql,
            chat=Mock(return_value=dict(status='chat_completed', usedSQL=False, answer='你好')))
        session._execute = Mock(return_value=dict(status='completed', sqlQuery='SELECT 1', sqlRows=[]))
        return session, model

    def test_missing_definition_returns_existing_public_choices_without_sql(self):
        s, _ = self.session([task(action='clarify', clarification='hole_definition')])
        result = s.run('WHU空洞比例是多少？')
        self.assertEqual(result['status'], 'needs_clarification')
        self.assertEqual([o['label'] for o in result['clarification']['options']],
                         ['标记样本占比', '面积比例均值'])
        self.assertIn('口径', result['answer'])
        self.assertEqual(result['clarification']['task']['regions'], ['whu'])
        self.assertNotIn('choiceValues', result['clarification'])
        s._execute.assert_not_called()

    def test_choice_survives_chat_and_restores_the_original_question(self):
        s, model = self.session([task(action='clarify', clarification='hole_definition'),
                                task(regions=['whu'], measure='hole_ratio_mean', filters=dict(reviewed=0,risk_level=None))])
        first = s.run('WHU未复核样本的空洞比例是多少？')
        greeting = s.run('你好')
        self.assertEqual(greeting['clarification']['id'], first['clarification']['id'])
        result = s.run('第二个')
        self.assertEqual(result['status'], 'completed')
        prompt = model.call_args.args[0]
        self.assertIn('WHU未复核样本的空洞比例是多少', prompt)
        self.assertIn('hole_ratio', prompt)
        self.assertEqual(result['clarificationResolution']['optionId'], 'option_2')
        self.assertIsNone(s.pending_question)

    def test_button_confirmation_rejects_a_model_that_changes_the_confirmed_metric(self):
        s, _ = self.session([task(action='clarify', clarification='hole_definition'),
                            task(regions=['whu'], measure='hole_mark_percent')])
        decision = s.run('WHU空洞比例')['clarification']
        result = s.run('面积比例均值', clarification_id=decision['id'], option_id='option_2')
        self.assertEqual(result['status'], 'semantic_rejected')
        self.assertIn('已确认', result['answer'])
        s._execute.assert_not_called()

    def test_selection_cannot_drop_original_explicit_filter_or_add_another_region(self):
        for question,wrong in [
            ('WHU未复核样本的空洞比例',task(regions=['whu'],measure='hole_ratio_mean')),
            ('WHU高风险样本的空洞比例',task(regions=['whu'],measure='hole_ratio_mean')),
            ('WHU空洞比例',task(regions=['whu','levir'],measure='hole_ratio_mean')),
        ]:
            with self.subTest(question=question):
                s,_=self.session([task(action='clarify',clarification='hole_definition'),wrong])
                decision=s.run(question)['clarification']
                result=s.run('面积比例均值',clarification_id=decision['id'],option_id='option_2')
                self.assertEqual(result['status'],'semantic_rejected',result)
                s._execute.assert_not_called()

    def test_region_choice_is_exact_for_a_singular_pending_question(self):
        s,_=self.session([task(action='clarify',clarification='region'),
                          task(action='clarify',clarification='hole_definition'),
                          task(regions=['whu','levir'],measure='hole_ratio_mean')])
        first=s.run('这个区域的空洞比例')['clarification']
        second=s.run('WHU',clarification_id=first['id'],option_id='option_1')['clarification']
        result=s.run('面积比例均值',clarification_id=second['id'],option_id='option_2')
        self.assertEqual(result['status'],'semantic_rejected',result)
        s._execute.assert_not_called()

    def test_risk_aggregate_is_not_a_sample_filter_after_region_choice(self):
        for measure in ('high_risk_percent','high_risk_count'):
            with self.subTest(measure=measure):
                s,_=self.session([task(action='clarify',clarification='region'),task(regions=['whu'],measure=measure)])
                first=s.run('这个区域高风险样本'+('占比' if measure=='high_risk_percent' else '数量')+'是多少？')['clarification']
                result=s.run('WHU',clarification_id=first['id'],option_id='option_1')
                self.assertEqual(result['status'],'completed',result)
                self.assertIsNone(s._execute.call_args.args[0]['filters']['risk_level'])

    def test_risk_percentage_cannot_filter_away_its_denominator(self):
        s,_=self.session([task(action='clarify',clarification='region'),
                          task(regions=['whu'],measure='high_risk_percent',filters=dict(reviewed=None,risk_level='high'))])
        first=s.run('这个区域高风险样本占比是多少？')['clarification']
        result=s.run('WHU',clarification_id=first['id'],option_id='option_1')
        self.assertEqual(result['status'],'semantic_rejected',result)
        s._execute.assert_not_called()

    def test_named_region_comparison_cannot_drop_one_region(self):
        s,_=self.session([task(action='clarify',clarification='hole_definition'),
                          task(regions=['whu'],measure='hole_ratio_mean',group_by='region')])
        first=s.run('WHU和LEVIR各个区域的空洞比例分别是多少？')['clarification']
        result=s.run('面积比例均值',clarification_id=first['id'],option_id='option_2')
        self.assertEqual(result['status'],'semantic_rejected',result)
        s._execute.assert_not_called()

    def test_other_regions_comparison_is_not_restricted_to_the_named_anchor(self):
        s,_=self.session([task(action='clarify',clarification='hole_definition'),
                          task(measure='hole_ratio_mean',group_by='region')])
        first=s.run('WHU与其他区域的空洞比例分别是多少？')['clarification']
        result=s.run('面积比例均值',clarification_id=first['id'],option_id='option_2')
        self.assertEqual(result['status'],'completed',result)

    def test_region_and_definition_can_be_clarified_in_successive_turns(self):
        s, model = self.session([task(action='clarify', clarification='region'),
                                task(action='clarify', clarification='hole_definition'),
                                task(regions=['whu'], measure='hole_ratio_mean')])
        first = s.run('这个区域空洞比例是多少？')['clarification']
        second = s.run('WHU', clarification_id=first['id'], option_id='option_1')['clarification']
        self.assertNotEqual(first['id'], second['id'])
        result = s.run('面积比例均值', clarification_id=second['id'], option_id='option_2')
        self.assertEqual(result['status'], 'completed')
        self.assertIn('WHU区域空洞比例是多少', model.call_args.args[0])
        self.assertIn('hole_ratio', model.call_args.args[0])

    def test_old_choice_is_rejected_after_session_or_scope_switch_without_model_or_sql(self):
        for switch in [dict(session_key='new'), dict(scope_region='whu')]:
            with self.subTest(switch=switch):
                s, model = self.session([task(action='clarify', clarification='region')])
                first = s.run('这个区域有多少样本？')['clarification']
                result = s.run('WHU', clarification_id=first['id'], option_id='option_1', **switch)
                self.assertEqual(result['source'], 'stale_clarification')
                self.assertFalse(result['usedSQL'])
                self.assertEqual(model.call_count, 1)
                s._execute.assert_not_called()

    def test_arbitrary_stale_button_never_falls_through_to_chat(self):
        s, model = self.session([])
        result = s.run('你好', clarification_id='expired', option_id='option_1')
        self.assertEqual(result['source'], 'stale_clarification')
        model.assert_not_called()
        s.chat.assert_not_called()

    def test_data_change_during_interpretation_invalidates_choice_before_sql(self):
        from model_sql import ModelSQL
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'data.json'
            raw=dict(regions=[dict(id='whu',short='WHU',items=[dict(id='w1',changedPx=10)])])
            source.write_text(json.dumps(raw))
            generate=Mock()
            sql=ModelSQL(source,Path(folder)/'history.db',generate=generate,model_id='test')
            def interpret(_):
                if interpret.calls==0:
                    interpret.calls+=1
                    return json.dumps(task(action='clarify',clarification='hole_definition'))
                raw['regions'][0]['items'][0]['changedPx']=20
                source.write_text(json.dumps(raw))
                return json.dumps(task(regions=['whu'],measure='hole_ratio_mean'))
            interpret.calls=0
            s=DirectSemanticQuerySession([dict(id='whu',label='WHU')],interpret,sql)
            decision=s.run('WHU空洞比例')['clarification']
            result=s.run('面积比例均值',clarification_id=decision['id'],option_id='option_2')
            self.assertEqual(result['source'],'stale_clarification')
            self.assertFalse(result['usedSQL']);generate.assert_not_called()
            self.assertIsNone(s._public_pending())

    def test_new_query_replaces_pending_and_missing_previous_task_does_not_invent_choices(self):
        s, _ = self.session([task(action='clarify', clarification='hole_definition'),
                            task(regions=['levir'], measure='sample_count'),
                            task(action='clarify', clarification='previous_task')])
        old = s.run('WHU空洞比例')['clarification']['id']
        self.assertEqual(s.run('换个问题，LEVIR多少样本？')['status'], 'completed')
        self.assertIsNone(s.pending_question)
        result = s.run('刚才选的几张求和')
        self.assertNotEqual(result['clarification']['id'], old)
        self.assertEqual(result['clarification']['options'], [])
        self.assertIn('样本', result['answer'])


if __name__ == '__main__':
    unittest.main()
