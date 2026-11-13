"""First-turn constraints must not depend on the SQL model noticing its own mistake."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_sql import ModelSQL


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.source=Path(self.temp.name)/'data.json'
        self.source.write_text(json.dumps({'regions':[
            {'id':'levir','short':'LEVIR','items':[{'id':'A','changedPx':10,'objectCount':0,'reviewRequired':True},
                {'id':'B','changedPx':21,'objectCount':3,'reviewLevel':'high'}]},
            {'id':'whu','short':'WHU','items':[{'id':'C','changedPx':30,'objectCount':0}]}]}),encoding='utf-8')

    def run_sql(self,question,sql,regions=('levir',),columns=('n',),repair=None):
        c=dict(action='query',question=question,regions=list(regions),columns=list(columns),order=[],
               limit=None,offset=0,per_group=False,calculations=[],missing=None)
        outputs=iter([json.dumps(c,ensure_ascii=False),sql,repair or sql])
        s=ModelSQL(self.source,Path(self.temp.name)/'q.sqlite',generate=lambda *a,**k:next(outputs),
                   model_id='fixture',contract_mode=True)
        return s.run(dict(question=question,sessionKey='s'))

    def test_or_cannot_bypass_explicit_region_on_first_turn(self):
        r=self.run_sql('只看LEVIR，有空洞或者对象数等于0的样本有多少张？',
            "SELECT COUNT(*) n FROM samples WHERE region_id='levir' AND has_holes=1 OR object_count=0")
        self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

    def test_region_error_is_repaired_by_model_not_replaced_with_template(self):
        r=self.run_sql('只看LEVIR，有空洞或者对象数等于0的样本有多少张？',
            "SELECT COUNT(*) n FROM samples WHERE region_id='levir' AND has_holes=1 OR object_count=0",
            repair="SELECT COUNT(*) n FROM samples WHERE region_id='levir' AND (has_holes=1 OR object_count=0)")
        self.assertEqual(r['sqlRows'],[{'n':1}]);self.assertEqual(r['llmCalls'],3)

    def test_unreviewed_is_not_needs_review(self):
        r=self.run_sql('LEVIR尚未复核的样本有多少张？',
            "SELECT COUNT(*) n FROM samples WHERE region_id='levir' AND needs_review=1")
        self.assertEqual(r['status'],'failed')

    def test_rounding_enforced_on_initial_question(self):
        r=self.run_sql('LEVIR平均变化像素是多少？保留一位小数，只返回均值。',
            "SELECT AVG(changed_px) mean FROM samples WHERE region_id='levir'",columns=('mean',))
        self.assertEqual(r['status'],'failed')

    def test_global_denominator_cannot_be_region_count(self):
        r=self.run_sql('LEVIR高风险样本数占所有区域全部样本总数的百分之几？只返回百分比，保留2位小数。',
            "SELECT ROUND(100.0*SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)/COUNT(*),2) pct FROM samples WHERE region_id='levir'",columns=('pct',))
        self.assertEqual(r['status'],'failed')

    def test_global_denominator_and_outer_region_both_preserved(self):
        r=self.run_sql('LEVIR高风险样本数占所有区域全部样本总数的百分之几？只返回百分比，保留2位小数。',
            "SELECT ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM samples),2) pct FROM samples s WHERE s.region_id='levir' AND s.review_level='high'",columns=('pct',))
        self.assertEqual(r['status'],'completed');self.assertEqual(r['sqlRows'],[{'pct':33.33}])

    def test_same_region_mean_subquery_is_not_mistaken_for_direct_avg(self):
        r=self.run_sql('LEVIR中高于本区域平均变化像素的样本有多少张？',
            "SELECT COUNT(*) n FROM samples s WHERE s.region_id='levir' AND s.changed_px>(SELECT AVG(changed_px) FROM samples WHERE region_id='levir')")
        self.assertEqual(r['status'],'completed')

    def test_multiple_named_regions_and_parenthesized_or(self):
        r=self.run_sql('LEVIR和WHU中有空洞或对象数为0的样本有多少张？',
            "SELECT COUNT(*) n FROM samples s WHERE s.region_id IN ('levir','whu') AND (has_holes=1 OR object_count=0)",regions=('levir','whu'))
        self.assertEqual(r['status'],'completed');self.assertEqual(r['sqlRows'],[{'n':2}])

    def test_unrelated_region_alias_does_not_prove_sample_scope(self):
        r=self.run_sql('只统计LEVIR的样本数量。',
            "SELECT COUNT(*) n FROM samples s JOIN regions r ON 1=1 WHERE r.region_id='levir'")
        self.assertEqual(r['status'],'failed')

    def test_cohort_followup_is_physically_bound_not_rejected_for_missing_region(self):
        # This path is covered with actual persisted state by test_query_population;
        # a qualified reviewed condition should independently remain valid.
        r=self.run_sql('LEVIR未复核样本数量是多少？',
            "SELECT COUNT(*) n FROM samples s WHERE s.region_id='levir' AND s.reviewed=0")
        self.assertEqual(r['status'],'completed')

    def test_forbidden_literal_is_not_erased_by_unknown_region_representation(self):
        for condition in ["region_id='whu'","region_id='levir' AND has_holes=1 OR region_id='whu'","region_id IN ('levir','whu')"]:
            with self.subTest(condition=condition):
                r=self.run_sql('只看LEVIR，有空洞的样本有多少张？','SELECT COUNT(*) n FROM samples WHERE '+condition)
                self.assertEqual(r['status'],'failed')

    def test_each_self_join_input_requires_its_own_scope(self):
        r=self.run_sql('只看LEVIR，样本有多少张？',
            "SELECT COUNT(*) n FROM samples s JOIN samples t ON t.sample_id='A' WHERE t.region_id='levir'")
        self.assertEqual(r['status'],'failed')

    def test_global_count_inside_arithmetic_is_not_a_global_denominator_proof(self):
        r=self.run_sql('LEVIR高风险样本数占所有区域全部样本总数的百分之几？保留2位小数。',
            "SELECT ROUND(100.0*COUNT(*)/((SELECT COUNT(*) FROM samples)-(SELECT COUNT(*) FROM samples WHERE region_id='whu')),2) pct FROM samples WHERE region_id='levir' AND review_level='high'",columns=('pct',))
        self.assertEqual(r['status'],'failed')

    def test_reviewed_scope_cannot_be_borrowed_from_other_join_alias(self):
        r=self.run_sql('尚未复核的样本有多少张？',
            "SELECT COUNT(*) n FROM samples s JOIN samples t ON t.sample_id='A' WHERE t.reviewed=0",regions=())
        self.assertEqual(r['status'],'failed')

    def test_global_denominator_with_single_selected_region_clarifies_before_generation(self):
        def forbidden(*a,**k):self.fail('Out-of-scope global denominator must be rejected before generation')
        s=ModelSQL(self.source,Path(self.temp.name)/'scope.sqlite',generate=forbidden,
                   model_id='fixture',contract_mode=True)
        r=s.run(dict(question='LEVIR高风险样本占全库样本总数百分之几？',sessionKey='s',scopeRegion='levir'))
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],0)

    def test_integer_risk_division_cannot_silently_return_zero(self):
        sql="SELECT region_id,SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)/COUNT(*)*100 pct FROM samples GROUP BY region_id"
        r=self.run_sql('各区域高风险占比是多少？',sql,regions=(),columns=('region_id','pct'))
        self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

    def test_floating_risk_division_is_repaired_before_execution(self):
        sql="SELECT region_id,SUM(CASE WHEN review_level='high' THEN 1 ELSE 0 END)/COUNT(*)*100 pct FROM samples GROUP BY region_id"
        r=self.run_sql('各区域高风险占比是多少？',sql,regions=(),columns=('region_id','pct'),repair=sql.replace('/COUNT(*)*100','*100.0/COUNT(*)'))
        self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['sqlRows'],[{'region_id':'levir','pct':50.0},{'region_id':'whu','pct':0.0}])
    def test_mandatory_inner_region_key_can_constrain_sample_source(self):
        sql="SELECT COUNT(*) n FROM samples s JOIN regions r ON s.region_id=r.region_id WHERE r.region_id='levir'"
        r=self.run_sql('只统计LEVIR样本总数',sql)
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'n':2}])
    def test_optional_or_join_equality_cannot_prove_region_constraint(self):
        sql="SELECT COUNT(*) n FROM samples s JOIN regions r ON s.region_id=r.region_id OR 1=1 WHERE r.region_id='levir'"
        r=self.run_sql('只统计LEVIR样本总数',sql)
        self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

if __name__=='__main__':unittest.main()
