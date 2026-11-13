import importlib,sys,json,sqlite3,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))

class SemanticProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'tools/query_semantic_protocol.py').exists(),'semantic protocol not implemented')
        self.p=importlib.import_module('query_semantic_protocol')
        self.ctx={'regions':[{'id':'whu','label':'WHU'},{'id':'levir','label':'LEVIR'}],'view_region':'levir','scope_region':None,'previous_task':None,'previous_selection':[],'previous_question':None}
    def test_count_and_region_group_compiler_returns_independent_expected_counts(self):
        t=self.p.task(regions=['whu'],measure='sample_count')
        with self.p.fixture_database(0) as db:self.assertEqual(self.p.execute_task(db,t,self.ctx),[{'sample_count':3}])
        t=self.p.task(regions=[],measure='sample_count',group_by='region')
        with self.p.fixture_database(0) as db:self.assertEqual(self.p.execute_task(db,t,self.ctx),[{'region_id':'levir','sample_count':2},{'region_id':'whu','sample_count':3}])
    def test_previous_selection_and_reference_population_remain_distinct(self):
        self.ctx['previous_selection']=['whu-a','whu-b']
        t=self.p.task(regions=['whu'],measure='above_region_mean_count',population='previous_selection',reference='region_all_mean')
        with self.p.fixture_database(0) as db:self.assertEqual(self.p.execute_task(db,t,self.ctx),[{'above_region_mean_count':0}])
        t['reference']='none'
        with self.assertRaises(self.p.SemanticError):self.p.validate_task(t,self.ctx)
    def test_risk_percent_denominator_keeps_unknown_and_other_risk_rows(self):
        t=self.p.task(regions=['whu'],measure='high_risk_percent')
        with self.p.fixture_database(0) as db:rows=self.p.execute_task(db,t,self.ctx)
        self.assertAlmostEqual(rows[0]['high_risk_percent'],200/3)
    def test_rank_interval_preserves_aggregation_and_pagination(self):
        t=self.p.task(regions=[],measure='sample_count',group_by='region',rank={'direction':'DESC','start':2,'end':2})
        with self.p.fixture_database(0) as db:self.assertEqual(self.p.execute_task(db,t,self.ctx),[{'region_id':'levir','sample_count':2}])
    def test_nonquery_and_illegal_fields_cannot_execute(self):
        for t in [self.p.task(action='chat'),self.p.task(action='clarify',clarification='region')]:
            self.p.validate_task(t,self.ctx)
            with self.p.fixture_database(0) as db,self.assertRaises(self.p.SemanticError):self.p.execute_task(db,t,self.ctx)
        t=self.p.task(regions=['whu'],measure='sample_count');t['sql']='DELETE FROM samples'
        with self.assertRaises(self.p.SemanticError):self.p.parse_task(json.dumps(t),self.ctx)
    def test_unknown_scope_and_missing_selection_are_rejected(self):
        for t in [self.p.task(regions=['outside'],measure='sample_count'),self.p.task(regions=['whu'],measure='sum_changed_px',population='previous_selection')]:
            with self.assertRaises(self.p.SemanticError):self.p.validate_task(t,self.ctx)
        self.ctx['scope_region']='levir'
        with self.assertRaises(self.p.SemanticError):self.p.validate_task(self.p.task(regions=['whu'],measure='sample_count'),self.ctx)
    def test_prompt_contains_context_and_original_question_but_never_expected_answer(self):
        text=self.p.make_prompt('你好吗，这里有几张图',self.ctx)
        self.assertIn('你好吗，这里有几张图',text);self.assertIn('levir',text)
        self.assertNotIn('expected_task',text);self.assertNotIn('SELECT COUNT',text)
    def test_requirement_renderer_preserves_explicit_scope_and_units(self):
        t=self.p.task(regions=['whu'],measure='mean_changed_ratio',filters={'reviewed':0,'risk_level':None})
        text=self.p.render_requirement(t,self.ctx)
        self.assertIn('WHU',text);self.assertIn('未复核',text);self.assertIn('均值',text)

if __name__=='__main__':unittest.main()
