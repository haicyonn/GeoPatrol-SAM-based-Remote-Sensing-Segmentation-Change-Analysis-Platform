import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation/dialogue_benchmark_20261103'))
try:
    from adjudicate import adjudicate
except ImportError:adjudicate=None

class AdjudicationTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(adjudicate,'post-freeze audit is not implemented')
    def entry(self,columns,rows,sql,expected=None,q='只返回样本数。'):
        return dict(question=q,expected=dict(route='sql',rows=expected or [{'count':3}]),response=dict(status='completed',usedSQL=True,sqlExecutionCount=1,sqlRows=rows,sqlColumns=columns,sqlQuery=sql))
    def test_alias_n_requires_count_expression(self):
        e=self.entry(['n'],[{'n':3}],'SELECT COUNT(*) AS n FROM samples')
        self.assertTrue(adjudicate(e)['correct'])
        e['response']['sqlQuery']='SELECT MAX(changed_px) AS n FROM samples'
        self.assertFalse(adjudicate(e)['correct'])
    def test_explicit_only_rejects_extra_column(self):
        e=self.entry(['count','region_id'],[{'count':3,'region_id':'whu'}],"SELECT COUNT(*) AS count,region_id FROM samples GROUP BY region_id")
        self.assertFalse(adjudicate(e)['correct'])
    def test_empty_rows_still_require_requested_schema(self):
        e=self.entry(['object_count'],[],'SELECT object_count FROM samples WHERE 0',expected=[])
        e['expected']['rows']=[];e['expected']['spec']={'fields':['sample','px']};e['question']='列出样本ID和变化像素。'
        self.assertFalse(adjudicate(e)['correct'])
    def test_alternative_alias_cannot_swap_roles(self):
        e=self.entry(['n','percentage'],[{'n':50,'percentage':2}],"SELECT COUNT(*) AS n,100.0 AS percentage FROM samples",expected=[{'high_count':2,'high_percent':50}])
        self.assertFalse(adjudicate(e)['correct'])

    def test_explicit_sql_rounding_uses_same_precision_without_hiding_wrong_units(self):
        e=self.entry(['percentage'],[{'percentage':33.33}],"SELECT ROUND(100.0*SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)/COUNT(*),2) AS percentage FROM samples",expected=[{'high_percent':100/3}],q='只返回高风险百分比。')
        self.assertTrue(adjudicate(e)['correct'])
        e['response']['sqlRows']=[{'percentage':0.33}]
        self.assertFalse(adjudicate(e)['correct'])

    def test_duplicate_region_labels_must_agree(self):
        e=self.entry(['region_id','region_name','sample_count'],[{'region_id':'whu','region_name':'WHU','sample_count':3}],"SELECT region_id,region_name,sample_count FROM regions",expected=[{'region':'whu','count':3}],q='返回区域和样本数。')
        self.assertTrue(adjudicate(e)['correct'])
        e['response']['sqlRows'][0]['region_name']='UAV'
        self.assertFalse(adjudicate(e)['correct'])

if __name__=='__main__':unittest.main()
