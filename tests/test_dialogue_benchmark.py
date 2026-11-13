import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation/dialogue_benchmark_20261103'))

try:
    from scoring import score, summarize
except ImportError:
    score=summarize=None


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(score,'independent benchmark scorer is not implemented')

    def response(self,rows):
        return dict(status='completed',questionRoute='sql',usedSQL=True,sqlExecutionCount=1,sqlRows=rows)

    def expect(self,rows,**kw):
        return dict(route='sql',rows=rows,ordered=True,**kw)

    def test_correct_alias_preserves_role(self):
        self.assertTrue(score(self.response([dict(region_id='whu',sample_count=3)]),self.expect([dict(region='whu',count=3)]))['correct'])

    def test_swapped_numeric_columns_fail(self):
        self.assertFalse(score(self.response([dict(high_risk_count=50,high_risk_percentage=2)]),self.expect([dict(high_count=2,high_percent=50)]))['correct'])

    def test_wrong_denominator_fails(self):
        self.assertFalse(score(self.response([dict(review_percentage=100)]),self.expect([dict(review_percent=50)]))['correct'])

    def test_rank_order_is_significant(self):
        self.assertFalse(score(self.response([dict(region_id='b',count=1),dict(region_id='a',count=2)]),self.expect([dict(region='a',count=2),dict(region='b',count=1)]))['correct'])

    def test_null_is_not_zero(self):
        self.assertFalse(score(self.response([dict(average_changed_ratio=0)]),self.expect([dict(mean_ratio=None)]))['correct'])

    def test_no_sql_means_no_sql_even_when_chat_status(self):
        r=dict(status='chat_completed',usedSQL=False,sqlExecutionCount=1)
        self.assertFalse(score(r,dict(route='chat'))['correct'])

    def test_clarification_requires_choices_and_no_execution(self):
        r=dict(status='needs_clarification',usedSQL=False,clarification={'options':[{'label':'A'},{'label':'B'}]})
        self.assertTrue(score(r,dict(route='clarify',choices=['A','B']))['correct'])
        r['sqlExecutionCount']=1
        self.assertFalse(score(r,dict(route='clarify',choices=['A','B']))['correct'])

    def test_unknown_alias_is_unscored_failure_not_guessed_by_value(self):
        self.assertFalse(score(self.response([dict(mystery=50)]),self.expect([dict(high_percent=50)]))['correct'])

    def test_timeouts_and_missing_turns_remain_in_denominator(self):
        s=summarize([dict(dialogue='a',correct=True),dict(dialogue='a',correct=False)],planned_turns=4,planned_dialogues=2)
        self.assertEqual(s['turns'],4)
        self.assertEqual(s['correct'],1)
        self.assertEqual(s['completed_dialogues'],0)


if __name__=='__main__':unittest.main()
