"""Grouped summaries must not be mistaken for within-group row ranking."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_sql import ModelSQL
from query_contract import parse_contract,unbounded_group_summary,planning_context

QUESTION='各区域高风险样本的数量和占比分别是多少？按占比降序排列。'
EXPR="SUM(CASE WHEN s.review_level='high' THEN 1 ELSE 0 END)"
SQL=f'SELECT r.region_name,{EXPR} high_count,100.0*{EXPR}/COUNT(*) pct FROM samples s JOIN regions r ON s.region_id=r.region_id GROUP BY r.region_id,r.region_name ORDER BY pct DESC'

def contract(**kw):
    c=dict(action='query',question=QUESTION,regions=[],columns=['region_name','high_count','pct'],
        order=[dict(expression='pct',direction='DESC')],limit=None,offset=0,per_group=False,calculations=[],missing=None)
    c.update(kw);return json.dumps(c,ensure_ascii=False)

class GroupSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.path=self.root/'data.json'
        self.path.write_text(json.dumps({'regions':[
            {'id':'whu','short':'WHU','items':[dict(id='W'+str(i),reviewLevel=l) for i,l in enumerate(['high','high','low','none','unknown'])]},
            {'id':'levir','short':'LEVIR','items':[dict(id='L'+str(i),reviewLevel=l) for i,l in enumerate(['high','low'])]},
            {'id':'gzcd','short':'GZ-CD','items':[dict(id='G'+str(i),reviewLevel='low') for i in range(4)]}]}),encoding='utf-8')
    def run_query(self,outputs,q=QUESTION):
        seq=iter(outputs);s=ModelSQL(self.path,self.root/'q.sqlite',generate=lambda *a,**kw:next(seq),model_id='fixture',contract_mode=True)
        return s.run(dict(question=q,sessionKey='s'))
    def test_false_group_ranking_flag_and_hallucinated_limit_are_normalized(self):
        r=self.run_query([contract(per_group=True,limit=2),SQL])
        self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['sqlRows'],[dict(region_name='LEVIR',high_count=1,pct=50.0),dict(region_name='WHU',high_count=2,pct=40.0),dict(region_name='GZ-CD',high_count=0,pct=0.0)])
        self.assertFalse(r['queryContract']['per_group']);self.assertIsNone(r['queryContract']['limit'])
        self.assertIn('group_summary_not_per_group_topn',r['queryContract']['normalizations'])
        self.assertIn('"per_group":true',r['attempts'][0]['modelGeneratedContract'].replace(' ',''))
    def test_false_group_flag_without_limit_is_fixed_before_schema_consistency_check(self):
        r=self.run_query([contract(per_group=True),SQL]);self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['llmCalls'],2)
    def test_no_arbitrary_top_limit_even_when_model_flag_is_false(self):
        r=self.run_query([contract(limit=1),SQL]);self.assertEqual(r['status'],'completed',r);self.assertEqual(len(r['sqlRows']),3)
    def test_counts_cannot_replace_requested_percentage_order(self):
        r=self.run_query([contract(),SQL.replace('ORDER BY pct DESC','ORDER BY high_count DESC'),SQL])
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'][0]['region_name'],'LEVIR')
        self.assertTrue(any(a['status']=='rejected' for a in r['attempts']))
    def test_summary_cannot_be_silently_truncated_by_sql(self):
        r=self.run_query([contract(),SQL+' LIMIT 2',SQL+' LIMIT 2'])
        self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])
    def test_true_each_region_topn_is_not_normalized(self):
        q='每个区域分别取变化像素最大的2个未复核样本，显示区域、样本ID和变化像素。'
        c=parse_contract(contract(question=q,columns=['region_id','sample_id'],order=[],per_group=True,limit=2),{'whu','levir','gzcd'},source_question=q)
        self.assertTrue(c['per_group']);self.assertEqual(c['limit'],2);self.assertNotIn('groupingMode',c)
    def test_explicit_global_topn_and_detailed_requests_are_not_coerced(self):
        for q in ['各区域高风险占比排名前3的区域','各区域分别取前二张样本','各区域高风险样本明细，按数量排序','每个区域变化像素最大的样本','各区域按风险等级统计数量','LEVIR高风险样本数占所有区域全部样本总数的百分之几？']:
            with self.subTest(q=q):self.assertFalse(unbounded_group_summary(q))
    def test_real_per_group_rank_still_requires_partitioned_bound(self):
        q='每个区域分别取前2张样本'
        c=contract(question=q,columns=['region_name','high_count','pct'],per_group=True,limit=2)
        r=self.run_query([c,SQL,SQL],q);self.assertEqual(r['status'],'failed')

    def ranked_query(self,direction='DESC'):
        q='每个区域分别取变化像素最大的2个未复核样本，显示区域、样本ID和变化像素。'
        c=contract(question=q,columns=['region_id','sample_id','changed_px'],per_group=True,limit=2,
            order=[dict(expression='changed_px',direction='DESC'),dict(expression='region_id',direction='ASC')])
        sql=('SELECT region_id,sample_id,changed_px FROM (SELECT region_id,sample_id,changed_px,'
             'ROW_NUMBER() OVER (PARTITION BY region_id ORDER BY changed_px '+direction+',sample_id ASC) rn '
             'FROM samples WHERE reviewed=0) ranked WHERE rn<=2 ORDER BY region_id,changed_px DESC,sample_id')
        return self.run_query([c,sql,sql],q)

    def test_group_selection_order_is_independent_of_outer_presentation(self):
        r=self.ranked_query();self.assertEqual(r['status'],'completed',r)
        self.assertEqual(len(r['sqlRows']),6)
        self.assertEqual(r['queryContract']['rankOrder'],[dict(expression='changed_px',direction='DESC')])

    def test_wrong_window_selection_direction_is_rejected(self):
        r=self.ranked_query('ASC');self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

    def test_computable_mean_is_retried_instead_of_asking_for_a_new_field(self):
        q='WHU中高于本区域平均变化像素的样本有多少张？'
        c=contract(question=q,regions=['whu'],columns=['n'],order=[])
        missing=json.dumps(dict(sql=None,clarification='缺少变化像素平均值的计算条件。'),ensure_ascii=False)
        sql="SELECT COUNT(*) n FROM samples WHERE region_id='whu' AND changed_px>(SELECT AVG(changed_px) FROM samples WHERE region_id='whu')"
        r=self.run_query([c,missing,sql],q)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[dict(n=0)])
        self.assertTrue(any(a['status']=='rejected' for a in r['attempts']))

    def test_missing_capture_date_is_still_a_real_clarification(self):
        q='WHU在2025年5月拍摄、且高于本区域平均变化像素的样本有多少张？'
        c=contract(question=q,regions=['whu'],columns=['n'],order=[])
        missing=json.dumps(dict(sql=None,clarification='缺少拍摄时间字段，无法计算指定日期的平均值。'),ensure_ascii=False)
        r=self.run_query([c,missing],q)
        self.assertEqual(r['status'],'needs_clarification',r);self.assertFalse(r['sqlRows'])

    def test_persisted_validator_annotations_are_not_fed_back_as_model_schema(self):
        stored=json.loads(contract(per_group=True,limit=2))
        stored.update(orderRequired=False,rankOrder=[dict(expression='changed_px',direction='DESC')],groupingMode='aggregate',normalizations=['internal'])
        original=dict(lastQuery=dict(queryContract=stored,last_list=dict(contract=stored,ordered_ids=['W1','W2'])))
        result=planning_context(original)
        self.assertEqual(result['lastQuery']['queryContract'],json.loads(contract(per_group=True,limit=2)))
        self.assertEqual(result['lastQuery']['last_list']['contract'],result['lastQuery']['queryContract'])
        self.assertEqual(result['lastQuery']['last_list']['ordered_ids'],['W1','W2'])
        self.assertIn('rankOrder',original['lastQuery']['queryContract'])

if __name__=='__main__':unittest.main()
