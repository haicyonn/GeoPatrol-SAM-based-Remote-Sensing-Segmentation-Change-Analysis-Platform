"""Independent regional operands, units, orientation and inherited filters."""
import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_sql import ModelSQL
from conversation_assistant import ConversationAssistant
from query_comparisons import plan,operand_shape,validate_operands,summarize,is_followup,ComparisonError

REGIONS=[dict(id='whu',short='WHU'),dict(id='levir',short='LEVIR')]
COUNT="SUM(CASE WHEN review_level = 'high' THEN 1 ELSE 0 END)"
QUESTION='whu相比levir区域，高风险样本少了多少?'

def sql(percentage=False,where=None):
    metric='100.0 * '+COUNT+' / COUNT(*)' if percentage else COUNT
    return 'SELECT region_id,'+metric+" AS metric_value FROM samples WHERE region_id IN ('whu','levir')"+(' AND '+where if where else '')+' GROUP BY region_id'

class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.options=[]
        self.source=self.root/'source.json'
        self.source.write_text(json.dumps(dict(regions=[dict(id='whu',short='WHU',items=[dict(id='W'+str(i),reviewLevel='high' if i<2 else 'low',changedPx=i*10) for i in range(5)]),dict(id='levir',short='LEVIR',items=[dict(id='L'+str(i),reviewLevel='high' if i<5 else 'low',changedPx=i*100) for i in range(10)])])),encoding='utf-8')
    def service(self,outputs):
        it=iter(outputs)
        def generate(p,**kw):self.options.append(kw);return next(it)
        return ConversationAssistant(ModelSQL(self.source,self.root/'sql.sqlite',generate=generate,model_id='fixture',contract_mode=True,query_update_adapter=True,reviewed_provider=lambda:{'W0','L0','L1'}),self.root/'chat.sqlite')
    def ask(self,s,q=QUESTION,**extra):return s.run(dict(question=q,sessionKey='compare')|extra)
    def test_explicit_risk_count_comparison_works_without_prior_history(self):
        r=self.ask(self.service([sql()]));self.assertEqual(r['status'],'completed',r)
        self.assertEqual((r['comparison']['leftValue'],r['comparison']['rightValue'],r['comparison']['value']),(2,5,-3));self.assertIn('少3个',r['answer']);self.assertEqual(r['llmCalls'],1)
        self.assertNotIn('use_query_adapter',self.options[0])
    def test_percentage_points_have_separate_denominators(self):
        r=self.ask(self.service([sql(True)]),'WHU比LEVIR高风险占比低多少？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['comparison']['value'],-10);self.assertEqual(r['comparison']['unit'],'个百分点')
    def test_relative_change_uses_the_right_region_as_denominator(self):
        r=self.ask(self.service([sql()]),'WHU比LEVIR高风险样本少了百分之几？')
        self.assertEqual(r['comparison']['value'],-60);self.assertEqual(r['comparison']['referenceDenominator'],5)
    def test_reverse_the_pair_preserves_the_operation(self):
        s=self.service([sql(),sql()]);self.ask(s,'WHU比LEVIR高风险样本少了百分之几？');r=self.ask(s,'反过来呢')
        self.assertEqual(r['comparison']['leftRegion'],'levir');self.assertEqual(r['comparison']['value'],150)
    def test_followup_switches_count_to_share_without_losing_pair(self):
        s=self.service([sql(),sql(True)]);self.ask(s);r=self.ask(s,'那占比呢')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['comparison']['unit'],'个百分点');self.assertEqual(r['comparison']['value'],-10)
    def test_wrong_scalar_comparison_cannot_silently_return_zero(self):
        bad="SELECT COUNT(*)-(SELECT COUNT(*)) difference FROM samples WHERE region_id IN ('whu','levir')"
        r=self.ask(self.service([bad,sql()]));self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlExecutionCount'],1);self.assertEqual(r['comparison']['value'],-3)
    def test_percentage_cannot_replace_an_explicit_count(self):
        r=self.ask(self.service([sql(True),sql()]));self.assertEqual(r['status'],'completed',r);self.assertEqual(r['comparison']['value'],-3)
    def test_regions_sample_count_is_not_a_risk_count_source(self):
        r=self.ask(self.service(['SELECT region_id,sample_count metric_value FROM regions',sql()]));self.assertEqual(r['status'],'completed',r)
    def test_false_missing_alias_is_repaired_internally(self):
        bad=json.dumps(dict(sql=None,clarification='缺少高风险样本数量字段，请提供。'),ensure_ascii=False)
        r=self.ask(self.service([bad,sql()]));self.assertEqual(r['status'],'completed',r);self.assertNotIn('clarification',r)
    def test_physical_scope_is_not_expanded(self):
        r=self.ask(self.service([]),scopeRegion='whu');self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],0);self.assertFalse(r['usedSQL'])
    def test_inherited_review_filter_is_not_dropped(self):
        q='各区域未复核高风险样本数量是多少？'
        c=json.dumps(dict(action='query',question=q,regions=[],columns=['region_id','n'],order=[],limit=None,offset=0,per_group=False,calculations=[],missing=None),ensure_ascii=False)
        first='SELECT region_id,'+COUNT+' AS n FROM samples WHERE reviewed=0 GROUP BY region_id'
        s=self.service([c,first,sql(),sql(where='reviewed = 0')]);self.assertEqual(self.ask(s,q)['status'],'completed')
        r=self.ask(s);self.assertEqual(r['status'],'completed',r);self.assertEqual(r['comparison']['value'],-2);self.assertIn('未复核',r['queryState']['filters'])
    def test_explicit_review_filter_is_bound_for_a_fresh_question(self):
        r=self.ask(self.service([sql(where='reviewed = 0')]),'WHU相比LEVIR，未复核高风险样本少多少？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['comparison']['value'],-2)
    def test_risk_is_a_filter_when_the_goal_is_a_numeric_mean(self):
        q='WHU相比LEVIR，高风险样本平均变化像素相差多少？'
        p=plan(q,None,REGIONS,None)
        self.assertEqual(p['measure'],dict(operation='avg',field='changed_px',risk=None));self.assertEqual(p['filterExpression'],"review_level = 'high'")
        query="SELECT region_id,AVG(changed_px) metric_value FROM samples WHERE region_id IN ('whu','levir') AND review_level='high' GROUP BY region_id"
        r=self.ask(self.service([query]),q);self.assertEqual(r['status'],'completed',r);self.assertEqual(r['comparison']['value'],-195)
    def test_new_risk_share_clears_the_old_risk_only_filter_from_denominator(self):
        history=dict(sql="SELECT region_id,AVG(changed_px) mean FROM samples WHERE review_level='high' GROUP BY region_id",queryContract=dict(question='高风险样本平均变化像素'))
        p=plan('WHU相比LEVIR，低风险占比相差多少？',history,REGIONS,None)
        self.assertIsNone(p['filterExpression']);self.assertEqual(p['measure']['risk'],'low')
    def test_reference_first_word_order_is_resolved(self):
        p=plan('和LEVIR相比，WHU高风险样本少多少？',None,REGIONS,None)
        self.assertEqual((p['leftRegion'],p['rightRegion']),('whu','levir'))
    def test_wrong_assumed_direction_is_corrected_by_actual_values(self):
        r=self.ask(self.service([sql()]),'LEVIR相比WHU，高风险样本少多少？')
        self.assertIn('多3个',r['answer']);self.assertEqual(r['comparison']['value'],3)
    def test_unknown_and_zero_denominators_do_not_produce_fake_numbers(self):
        p=plan('WHU比LEVIR高风险样本少了百分之几',None,REGIONS,None)
        answer,record=summarize([dict(region_id='whu',metric_value=2),dict(region_id='levir',metric_value=0)],p,REGIONS)
        self.assertIsNone(record['value']);self.assertIn('基准值为0',answer)
        p['measure']=dict(operation='avg',field='hole_ratio',risk=None);p['operation']='difference'
        answer,record=summarize([dict(region_id='whu',metric_value=None),dict(region_id='levir',metric_value=.1)],p,REGIONS)
        self.assertIsNone(record['value']);self.assertIn('无法计算',answer)
    def test_missing_count_group_means_zero_not_a_missing_schema_field(self):
        p=plan(QUESTION,None,REGIONS,None);answer,record=summarize([dict(region_id='levir',metric_value=5)],p,REGIONS)
        self.assertEqual(record['leftValue'],0);self.assertEqual(record['value'],-5)
    def test_arithmetic_ratio_is_different_from_a_percentage_point_difference(self):
        p=plan('WHU的高风险样本数量是LEVIR的多少倍？',None,REGIONS,None)
        _,r=summarize([dict(region_id='whu',metric_value=2),dict(region_id='levir',metric_value=5)],p,REGIONS)
        self.assertAlmostEqual(r['value'],.4);self.assertEqual(r['unit'],'倍')
    def test_unrelated_chat_is_not_a_comparison_followup(self):
        p=plan(QUESTION,None,REGIONS,None)
        self.assertFalse(is_followup('那你开心吗',p));self.assertFalse(is_followup('解释占比是什么意思',p))
    def test_ambiguous_measure_offers_bound_choices_and_resumes_original_pair(self):
        q='各区域高风险样本的数量和占比分别是多少？按占比降序。'
        first='SELECT region_id,'+COUNT+' AS n,100.0*'+COUNT+'/COUNT(*) AS pct FROM samples GROUP BY region_id ORDER BY pct DESC'
        c=json.dumps(dict(action='query',question=q,regions=[],columns=['region_id','n','pct'],order=[dict(expression='pct',direction='DESC')],limit=None,offset=0,per_group=False,calculations=[],missing=None),ensure_ascii=False)
        s=self.service([c,first,sql()]);self.assertEqual(self.ask(s,q)['status'],'completed')
        pending=self.ask(s,'WHU比LEVIR相差多少？')
        self.assertEqual(pending['status'],'needs_clarification');self.assertEqual(pending['llmCalls'],0)
        choices=pending['clarification'];self.assertEqual(choices['slot'],'comparison_metric');self.assertEqual(len(choices['options']),2)
        option=next(o for o in choices['options'] if '数量' in o['label'])
        r=self.ask(s,option['label'],clarificationId=choices['id'],optionId=option['id'])
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['comparison']['value'],-3)

if __name__=='__main__':unittest.main()
