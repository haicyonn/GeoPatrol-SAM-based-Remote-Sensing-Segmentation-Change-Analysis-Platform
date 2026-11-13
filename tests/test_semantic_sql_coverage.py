import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))

class SemanticSQLCoverageTests(unittest.TestCase):
    def test_extra_denominator_is_allowed_when_one_consistent_projection_matches_all_databases(self):
        import semantic_sql_coverage as s
        actual=[[dict(region='x',n=4,rate=25.0)],[dict(region='x',n=8,rate=50.0)]]
        expected=[[dict(region_id='x',high_risk_percent=25.0)],[dict(region_id='x',high_risk_percent=50.0)]]
        self.assertTrue(s.contains_requested_projection(actual,expected))
    def test_cannot_choose_a_different_column_for_each_database(self):
        import semantic_sql_coverage as s
        actual=[[dict(a=2,b=8)],[dict(a=3,b=9)]]
        expected=[[dict(sample_count=2)],[dict(sample_count=9)]]
        self.assertFalse(s.contains_requested_projection(actual,expected))
if __name__=='__main__':unittest.main()
