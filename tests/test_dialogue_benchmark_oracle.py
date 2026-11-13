import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation/dialogue_benchmark_20261103'))
from oracle import compute,contrast
from cases import build

class OracleTests(unittest.TestCase):
    def test_review_denominator_excludes_not_required_but_reviewed(self):
        result=compute(contrast(),dict(kind='scalar',regions=['whu'],metrics=['review_percent']))
        self.assertEqual(result,[{'review_percent':50}])
    def test_null_hole_does_not_become_zero(self):
        self.assertEqual(compute(contrast(),dict(kind='scalar',regions=['clcd'],metrics=['mean_hole'])),[{'mean_hole':None}])
    def test_target_and_reference_different_sets(self):
        d={'regions':[dict(id='whu',items=[dict(id=str(i),changedPx=p) for i,p in enumerate([1,60,70,100])])]}
        s=dict(kind='selected',selection=dict(kind='samples',regions=['whu'],sort='px',limit=2,fields=['sample','px']),reference={'regions':['whu']},metrics=['count'])
        self.assertEqual(compute(d,s),[{'count':2}])
    def test_plan_size_and_fixture_separation(self):
        cs=build({'real':contrast(),'contrast':contrast()})
        self.assertEqual(len(cs),50);self.assertEqual(sum(c['dataset']=='real' for c in cs),40)
        self.assertEqual(len({c['id'] for c in cs}),50)

if __name__=='__main__':unittest.main()
