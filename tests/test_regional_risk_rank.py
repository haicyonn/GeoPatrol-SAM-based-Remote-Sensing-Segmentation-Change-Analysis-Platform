"""Separate the region-selection metric from the reported risk metric."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_sql import ModelSQL
from query_semantics import regional_total_rank

QUESTION='我比较想询问样本最多的区域的小风险占比'
REWRITE='查询样本最多的区域的低风险占比。'
SQL="""SELECT region_id,COUNT(*) AS total,
ROUND(100.0*SUM(CASE WHEN review_level='low' THEN 1 ELSE 0 END)/COUNT(*),2) AS pct
FROM samples GROUP BY region_id ORDER BY total DESC,region_id ASC LIMIT 1"""

def contract(question=REWRITE,**kw):
    c=dict(action='query',question=question,regions=[],columns=['region_id','total','pct'],
           order=[dict(expression='total',direction='DESC'),dict(expression='region_id',direction='ASC')],
           limit=1,offset=0,per_group=False,calculations=['先按区域全部样本总数选区域；low数量除以该区域全部样本数'],missing=None)
    c.update(kw)
    return json.dumps(c,ensure_ascii=False)

class RegionalRiskRankTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'data.json'
        # WHU has more TOTAL samples but LEVIR has more LOW samples. none and
        # unknown must stay in the denominator and never become low risk.
        self.source.write_text(json.dumps({'regions':[
            {'id':'whu','short':'WHU','items':[dict(id='W'+str(i),reviewLevel=level) for i,level in enumerate(['low','high','none','unknown'])]},
            {'id':'levir','short':'LEVIR','items':[dict(id='L'+str(i),reviewLevel=level) for i,level in enumerate(['low','low','medium'])]}]}),encoding='utf-8')

    def service(self,outputs):
        seq=iter(outputs)
        return ModelSQL(self.source,self.root/'q.sqlite',generate=lambda *a,**kw:next(seq),model_id='fixture',contract_mode=True)

    def ask(self,outputs,q=QUESTION):return self.service(outputs).run(dict(question=q,sessionKey='s'))

    def test_total_rank_and_low_risk_percentage_are_distinct(self):
        r=self.ask([contract(),SQL])
        self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['sqlRows'],[dict(region_id='whu',total=4,pct=25.0)])
        self.assertIn('regional_total_rank_and_risk_share',r['semanticChecks'])

    def test_wrong_rewrite_is_rejected_and_model_repairs_it(self):
        r=self.ask([contract('哪个区域的小风险样本最多，计算占比？'),contract(),SQL])
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['llmCalls'],3)
        self.assertEqual(r['attempts'][0]['status'],'rejected')

    def test_repeated_wrong_interpretation_never_executes(self):
        bad=contract('低风险样本最多的区域及其占比')
        r=self.ask([bad,bad]);self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

    def test_derived_alias_is_not_missing_input(self):
        bad=contract(action='clarify',missing=dict(kind='unavailable_field',field='low_risk_samples',question='缺少low_risk_samples字段'))
        r=self.ask([bad,contract(),SQL]);self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['attempts'][0]['status'],'rejected')

    def test_sql_false_missing_alias_and_region_are_repaired(self):
        for reason in ["缺少必要的字段 'low_risk_samples'，请提供该字段",'请明确哪个区域，提供具体的区域ID或名称。']:
            with self.subTest(reason=reason):
                r=self.ask([contract(),json.dumps(dict(sql=None,clarification=reason)),SQL])
                self.assertEqual(r['status'],'completed',r);self.assertEqual(r['llmCalls'],3)

    def test_sql_cannot_rank_by_conditional_count(self):
        wrong=SQL.replace('COUNT(*) AS total',"SUM(CASE WHEN review_level='low' THEN 1 ELSE 0 END) AS total")
        r=self.ask([contract(),wrong,SQL]);self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['attempts'][1]['status'],'rejected')
        self.assertEqual(r['sqlRows'][0]['region_id'],'whu')

    def test_reversed_equijoin_and_real_cast_are_valid(self):
        sql=SQL.replace('region_id,COUNT(*)','r.region_id,COUNT(*)').replace('FROM samples GROUP BY region_id',
            'FROM regions r JOIN samples s ON r.region_id=s.region_id GROUP BY r.region_id').replace('total DESC,region_id','total DESC,r.region_id')
        sql=sql.replace("100.0*SUM(CASE WHEN review_level='low' THEN 1 ELSE 0 END)","CAST(SUM(CASE WHEN review_level='low' THEN 1 ELSE 0 END) AS REAL)*100")
        r=self.ask([contract(),sql]);self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['sqlRows'][0]['pct'],25.0)

    def test_undefined_total_count_sort_alias_triggers_plan_repair(self):
        bad=contract(order=[dict(expression='sample_count',direction='DESC')])
        r=self.ask([bad,contract(),SQL]);self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['attempts'][0]['status'],'rejected')

    def test_count_of_non_null_sample_ids_matches_star_with_inner_join(self):
        sql=SQL.replace('COUNT(*)','COUNT(s.sample_id)').replace('region_id,COUNT','r.region_id,COUNT').replace('FROM samples GROUP BY region_id',
            'FROM regions r JOIN samples s ON r.region_id=s.region_id GROUP BY r.region_id').replace('total DESC,region_id','total DESC,r.region_id')
        c=contract(order=[dict(expression='COUNT(*)',direction='DESC'),dict(expression='region_id',direction='ASC')])
        r=self.ask([c,sql]);self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['sqlRows'][0]['pct'],25.0)

    def test_missing_top_one_is_not_accepted(self):
        r=self.ask([contract(limit=None),contract(),SQL]);self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['attempts'][0]['status'],'rejected')

    def test_no_risk_prefilter_wrong_denominator_or_risk_merging(self):
        for wrong in [SQL.replace('GROUP BY',"WHERE review_level='low' GROUP BY"),
                      SQL.replace("review_level='low'","review_level IN ('low','none')"),
                      SQL.replace('/COUNT(*)','/SUM(CASE WHEN review_level!=\'unknown\' THEN 1 ELSE 0 END)'),
                      SQL.replace('100.0*SUM','100.0*100.0*SUM')]:
            with self.subTest(wrong=wrong):
                r=self.ask([contract(),wrong,wrong]);self.assertEqual(r['status'],'failed',r)

    def test_legitimate_risk_rank_is_not_reclassified_as_total_rank(self):
        self.assertIsNone(regional_total_rank('低风险样本最多的区域的占比'))
        self.assertIsNone(regional_total_rank('小风险的样本最多的区域'))
        q='低风险样本最多的区域的小风险占比'
        wrong_for_total=SQL.replace('COUNT(*) AS total',"SUM(CASE WHEN review_level='low' THEN 1 ELSE 0 END) AS total")
        r=self.ask([contract(q),wrong_for_total],q)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'][0]['region_id'],'levir')

    def test_genuinely_unavailable_date_still_clarifies(self):
        bad=contract(action='clarify',missing=dict(kind='unavailable_field',field='captured_at',question='没有拍摄日期，不能统计最近7天。'))
        r=self.ask([bad],QUESTION+'，只看最近7天拍摄的')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],1)

if __name__=='__main__':unittest.main()
