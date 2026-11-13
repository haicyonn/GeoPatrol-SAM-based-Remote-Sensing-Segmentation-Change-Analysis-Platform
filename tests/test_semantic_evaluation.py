import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import evaluate_semantic_conversations as e
from query_semantic_protocol import fixture_database

class SemanticEvaluationTests(unittest.TestCase):
    def test_rank_fixture_avoids_unspecified_tie_breaks_in_user_dialogue(self):
        for variant in range(3):
            with fixture_database(variant,e.REGIONS) as db:values=[r[0] for r in db.execute('SELECT AVG(hole_ratio) FROM samples GROUP BY region_id')]
            self.assertEqual(len(values),len(set(values)))
    def test_contrast_data_distinguishes_reviewed_from_high_risk(self):
        from query_semantic_protocol import task,execute_task
        matches=[]
        for variant in range(3):
            with fixture_database(variant,e.REGIONS) as db:
                expected=execute_task(db,task(measure='high_risk_percent',group_by='region'),dict(regions=e.REGIONS,view_region=None,scope_region=None,previous_task=None,previous_question=None,previous_selection=[]))
                wrong=[dict(r) for r in db.execute('SELECT region_id,100.0*SUM(reviewed)/COUNT(*) high_risk_percent FROM samples GROUP BY region_id ORDER BY region_id')]
                matches.append(expected==wrong)
        self.assertFalse(any(matches))
    def test_open_ended_rank_renderer_preserves_start(self):
        from query_semantic_protocol import task,render_requirement
        c=dict(regions=e.REGIONS,view_region=None,scope_region=None,previous_task=None,previous_question=None,previous_selection=[])
        q=render_requirement(task(measure='sample_count',group_by='region',rank=dict(direction='DESC',start=2,end=None)),c)
        self.assertIn('第2条',q)
    def test_aliases_cannot_reorder_identity_and_metric_in_score(self):
        self.assertEqual(e.table_values([dict(region='whu',n=3)],None),e.table_values([dict(region_id='whu',sample_count=3)],None))
    def test_wrong_clarification_reason_does_not_pass(self):
        self.assertFalse(e.clarification_matches(dict(status='needs_clarification',semanticTask=dict(clarification='region')),'hole_definition'))
        self.assertTrue(e.clarification_matches(dict(status='needs_clarification',semanticTask=dict(clarification='region')),'region'))
    def test_public_snapshot_preserves_every_oracle_column(self):
        from model_sql import PublicQueryData
        with tempfile.TemporaryDirectory() as temp:
            for variant in range(3):
                path=Path(temp)/'fixture.json';e.export_fixture(path,variant)
                actual=PublicQueryData(path).execute('SELECT sample_id,region_id,changed_px,changed_ratio,review_level,reviewed,hole_ratio,has_holes FROM samples ORDER BY sample_id')['rows']
                with fixture_database(variant,e.REGIONS) as db:expected=[dict(r) for r in db.execute('SELECT * FROM samples ORDER BY sample_id')]
                self.assertEqual(actual,expected)
if __name__=='__main__':unittest.main()
