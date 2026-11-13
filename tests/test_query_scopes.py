import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_sql import ModelSQL
from query_scopes import parse_scopes,prepare,validate_and_expand


def contract(**kw):
    c=dict(action='query',question='范围比较',regions=['whu'],columns=['n'],order=[],limit=None,offset=0,per_group=False,calculations=[],missing=None)
    c.update(kw)
    return json.dumps(c,ensure_ascii=False)


def scopes(region='whu',mode='previous',**kw):
    value=dict(selection=dict(mode=mode,region=region,sort=None,limit=None,filters=[]),
               reference=dict(scope='region',region=region,metric='changed_px',aggregate='avg'),
               comparison=dict(metric='changed_px',operator='gt',output='count'))
    for k,v in kw.items():value[k].update(v)
    return value


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'data.json';self.prompts=[]
        self.source.write_text(json.dumps({'regions':[
            {'id':'whu','short':'WHU','items':[dict(id='W'+str(i),changedPx=110-10*i if i<=10 else 0,objectCount=i) for i in range(1,21)]},
            {'id':'levir','short':'LEVIR','items':[dict(id='L'+str(i),changedPx=1000 if i==1 else 20-i,objectCount=i) for i in range(1,21)]}]}),encoding='utf-8')

    def service(self,outputs):
        iterator=iter(outputs)
        def generate(prompt,**kw):
            self.prompts.append(prompt)
            return {'text':next(iterator),'finishReason':'stop'}
        return ModelSQL(self.source,self.root/'history.sqlite',generate=generate,model_id='test',contract_mode=True)

    def first(self,s,where="region_id='whu'"):
        return self.ask(s,'找出WHU变化像素最多的10张，只返回样本ID和变化像素。')

    def initial(self,where="region_id='whu'"):
        return [contract(question='WHU前10张',columns=['sample_id','changed_px'],order=[dict(expression='changed_px',direction='DESC'),dict(expression='sample_id',direction='ASC')],limit=10),
                'SELECT sample_id,changed_px FROM samples WHERE '+where+' ORDER BY changed_px DESC,sample_id ASC LIMIT 10']

    def ask(self,s,q,**kw):return s.run(dict(question=q,sessionKey='scope-tests',_authenticatedUser='alice',**kw))

    def compare(self,**kw):
        return [contract(scopes=scopes(**kw)), 'SELECT COUNT(*) n FROM selected_samples s WHERE s.changed_px>(SELECT AVG(changed_px) FROM reference_samples)']

    def test_two_populations_and_region_reselection(self):
        s=self.service(self.initial()+self.compare()+[
            contract(regions=['levir'],scopes=scopes('levir','reselect')),
            'WITH b AS (SELECT AVG(changed_px) mu FROM reference_samples) SELECT COUNT(*) n FROM selected_samples WHERE changed_px>(SELECT mu FROM b)'])
        self.assertEqual(self.first(s)['status'],'completed')
        r=self.ask(s,'这10张里面，高于WHU全部样本平均变化像素的有多少张？')
        self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['sqlRows'],[{'n':8}])
        self.assertIn('scoped_populations',r['programChanges'])
        r=self.ask(s,'换成LEVIR，也取前10张，与LEVIR自己的平均值比较。')
        self.assertEqual(r['status'],'completed',r)
        self.assertEqual(r['sqlRows'],[{'n':1}])
        self.assertEqual(r['population']['region_ids'],['levir'])
        self.assertTrue(all(x.startswith('L') for x in r['population']['filters']['values']))

    def test_paraphrase_does_not_change_population_behavior(self):
        for q in ['这10张里面，高于WHU全部样本平均变化像素的有多少张？','这10张与WHU全部样本平均变化像素相比，有多少张更高？']:
            with self.subTest(q=q):
                s=self.service(self.initial()+self.compare());self.first(s)
                r=self.ask(s,q)
                self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'n':8}])

    def test_wrong_reference_role_is_rejected_and_repaired(self):
        p=contract(scopes=scopes())
        bad='SELECT COUNT(*) n FROM selected_samples s WHERE s.changed_px>(SELECT AVG(changed_px) FROM selected_samples)'
        good='SELECT COUNT(*) n FROM selected_samples s WHERE s.changed_px>(SELECT AVG(changed_px) FROM reference_samples)'
        s=self.service(self.initial()+[p,bad,good]);self.first(s)
        r=self.ask(s,'这10张里面，高于WHU全部样本平均变化像素的有多少张？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'n':8}])
        self.assertEqual(r['llmCalls'],3)
        self.assertTrue(any(x['status']=='rejected' for x in r['attempts']))

    def test_region_change_preserves_existing_filter_and_ranking(self):
        value=scopes('levir','reselect')
        s=self.service(self.initial("region_id='whu' AND object_count>=5")+self.compare()+[
            contract(regions=['levir'],scopes=value),self.compare()[1]])
        self.first(s)
        self.ask(s,'这10张与WHU全部样本平均值相比有多少张更高？')
        r=self.ask(s,'换成LEVIR，其他筛选和排名保持不变，与LEVIR自己的均值比较。')
        self.assertEqual(r['status'],'completed',r)
        self.assertIn('object_count >= 5',r['sqlQuery'])
        self.assertEqual(r['sqlRows'],[{'n':0}])

    def test_explicit_filter_replace_then_remove(self):
        replacement=scopes(mode='reselect',selection={'filters':[dict(action='set',field='object_count',operator='gte',value=9)]})
        removal=scopes(mode='reselect',selection={'filters':[dict(action='remove',field='object_count',operator=None,value=None)]})
        s=self.service(self.initial("region_id='whu' AND object_count>=5")+self.compare()+[
            contract(scopes=replacement),self.compare()[1],contract(scopes=removal),self.compare()[1]])
        self.first(s);self.ask(s,'这10张与WHU全部样本平均值相比有多少张更高？')
        r=self.ask(s,'把对象数下限改成9，仍取前10张，与WHU全部均值比较。')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'n':0}])
        r=self.ask(s,'取消对象数量限制，其他不变，仍与WHU全部均值比较。')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'n':8}])

    def test_stale_data_and_other_user_do_not_reuse_ids(self):
        s=self.service(self.initial());self.first(s)
        r=s.run(dict(question='这10张与WHU全部样本均值比较有多少张更高？',sessionKey='scope-tests',_authenticatedUser='bob'))
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],0)
        d=json.loads(self.source.read_text());d['regions'][0]['items'][0]['changedPx']=999
        self.source.write_text(json.dumps(d),encoding='utf-8')
        r=self.ask(s,'这10张与WHU全部样本均值比较有多少张更高？')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],0)

    def test_physical_region_restriction_is_retained(self):
        s=self.service(self.initial());self.ask(s,'WHU变化像素最多的10张',scopeRegion='whu')
        r=self.ask(s,'这10张与所有区域全部样本的平均值相比有多少张更高？',scopeRegion='whu')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],0)

    def test_percentage_uses_distinct_reference_count(self):
        value=scopes(reference={'aggregate':'count','metric':None},comparison={'metric':None,'operator':'share','output':'percentage'})
        s=self.service(self.initial()+[contract(columns=['pct'],scopes=value),
            'SELECT 100.0*COUNT(*)/NULLIF((SELECT COUNT(*) FROM reference_samples),0) pct FROM selected_samples'])
        self.first(s);r=self.ask(s,'这10张占WHU全部样本的百分之几？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'pct':50.0}])

    def test_fraction_of_selected_samples_above_mean(self):
        value=scopes(comparison={'output':'percentage'})
        s=self.service(self.initial()+[contract(columns=['pct'],scopes=value),
            'SELECT 100.0*SUM(CASE WHEN changed_px>(SELECT AVG(changed_px) FROM reference_samples) THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0) pct FROM selected_samples'])
        self.first(s);r=self.ask(s,'这10张中，高于WHU全部样本平均变化像素的占比是多少？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'pct':80.0}])

    def test_empty_and_unknown_comparisons_do_not_fabricate_values(self):
        for where in ["region_id='whu' AND changed_px>10000", "region_id='whu' AND object_count<0"]:
            with self.subTest(where=where):
                s=self.service(self.initial(where)+self.compare());self.first(s)
                r=self.ask(s,'这些样本与WHU全部样本均值比较，有多少张更高？')
                self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'n':0}])

    def test_unknown_reference_is_not_reported_as_zero_matches(self):
        d=json.loads(self.source.read_text())
        for row in d['regions'][1]['items']:row['changedPx']=None
        self.source.write_text(json.dumps(d),encoding='utf-8')
        value=scopes(reference={'region':'levir'})
        s=self.service(self.initial()+[contract(regions=['whu','levir'],scopes=value)])
        self.first(s)
        r=self.ask(s,'这10张高于LEVIR全部样本平均变化像素的有多少？')
        self.assertEqual(r['status'],'needs_clarification',r)
        self.assertEqual(r['clarificationSource'],'reference_data_guard')
        self.assertEqual(r['sqlRows'],[])

    def test_named_region_without_word_all_uses_separate_reference(self):
        s=self.service(self.initial()+self.compare());self.first(s)
        r=self.ask(s,'这10张里高于WHU平均变化像素的有多少张？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'n':8}])

    def test_reselection_survives_restart_and_updates_displayed_reference(self):
        s=self.service(self.initial()+self.compare()+[
            contract(regions=['levir'],scopes=scopes('levir','reselect')),self.compare()[1]])
        self.first(s);self.ask(s,'这10张与WHU全部样本平均值相比有多少张更高？')
        r=self.ask(s,'换成LEVIR，与LEVIR自己的均值比较，其他不变。')
        self.assertEqual(r['status'],'completed',r)
        ids=r['selectionSampleIds'];self.assertEqual(len(ids),10)
        restored=self.service([contract(regions=['levir'],columns=['changed_px']),
                              "SELECT changed_px FROM samples WHERE sample_id='"+ids[1]+"'"])
        r=self.ask(restored,'刚才第二张的变化像素是多少？只返回变化像素。')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['resolvedReference']['sampleId'],ids[1])

    def test_sort_and_limit_change_keeps_other_filters(self):
        value=scopes(mode='reselect',selection={'sort':dict(field='object_count',direction='DESC'),'limit':3})
        sql='SELECT COUNT(*) n FROM selected_samples s WHERE s.changed_px>(SELECT AVG(changed_px) FROM reference_samples)'
        s=self.service(self.initial("region_id='whu' AND object_count>=5")+self.compare()+[contract(scopes=value),sql])
        self.first(s);self.ask(s,'这10张与WHU全部样本平均值相比有多少张更高？')
        r=self.ask(s,'排序改为对象数从大到小，只取前3张，其余条件不变，仍与WHU全部变化像素均值比较。')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'n':0}])
        self.assertEqual(r['population']['sample_count'],3)
        self.assertIn('object_count >= 5',r['sqlQuery'])

    def test_wrong_own_region_and_unrequested_filter_edits_are_rejected(self):
        for value in [scopes('levir','reselect',reference={'region':'whu'}),
                      scopes('levir','reselect',selection={'filters':[dict(action='remove',field='object_count',operator=None,value=None)]})]:
            with self.subTest(value=value):
                s=self.service(self.initial()+self.compare()+[contract(regions=['levir'],scopes=value)]*1)
                self.first(s);self.ask(s,'这10张与WHU全部样本平均值相比有多少张更高？')
                r=self.ask(s,'换成LEVIR，其他条件保持不变，与LEVIR自己的均值比较。')
                self.assertNotEqual(r['status'],'completed');self.assertEqual(r['sqlRows'],[])

    def test_invalid_sql_shapes_do_not_bypass_independent_scopes(self):
        ids=','.join("'W%d'"%i for i in range(1,11))
        p=dict(supported=True,kind='displayed_samples',region_ids=['whu'],selector_sql='SELECT sample_id FROM samples WHERE sample_id IN ('+ids+')')
        prepared=prepare(parse_scopes(scopes(),{'whu','levir'}),{'population':p},None,'这10张与WHU全部均值比较')
        bad_queries=[
          'SELECT COUNT(*) n FROM samples WHERE changed_px>(SELECT AVG(changed_px) FROM reference_samples)',
          'SELECT COUNT(*) n FROM selected_samples WHERE changed_px>(SELECT AVG(changed_px) FROM selected_samples)',
          'SELECT COUNT(*) n FROM selected_samples WHERE changed_px>(SELECT AVG(changed_px) FROM reference_samples WHERE object_count>3)',
          'SELECT COUNT(*) n FROM selected_samples WHERE changed_px>(SELECT AVG(changed_px) FROM reference_samples) OR 1=1',
          'SELECT COUNT(*)+100 n FROM selected_samples WHERE changed_px>(SELECT AVG(changed_px) FROM reference_samples)',
          'WITH x AS (SELECT object_count AS changed_px FROM selected_samples) SELECT COUNT(*) n FROM x WHERE changed_px>(SELECT AVG(changed_px) FROM reference_samples)',
          'WITH selected_samples AS (SELECT * FROM reference_samples) SELECT COUNT(*) n FROM selected_samples WHERE changed_px>(SELECT AVG(changed_px) FROM reference_samples)',
          'SELECT COUNT(*) n FROM selected_samples WHERE changed_px>(SELECT AVG(changed_px) FROM reference_samples); DELETE FROM samples',
        ]
        for sql in bad_queries:
            with self.subTest(sql=sql):
                with self.assertRaises(ValueError):validate_and_expand(sql,prepared)

    def test_compact_model_plan_is_compiled_without_repeating_ids(self):
        value=scopes();value['clarification']=None
        s=self.service(self.initial()+[json.dumps(value),self.compare()[1]])
        self.first(s);r=self.ask(s,'这10张里高于WHU全部样本平均变化像素的有多少张？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'n':8}])
        self.assertNotIn("'W1'",self.prompts[2])
        self.assertEqual(r['sqlColumns'],['n'])

    def test_redundant_proven_region_filters_are_safe(self):
        sql="SELECT COUNT(*) n FROM selected_samples s WHERE s.region_id='whu' AND (SELECT AVG(changed_px) FROM reference_samples r WHERE r.region_id='whu')<s.changed_px"
        s=self.service(self.initial()+[contract(scopes=scopes()),sql]);self.first(s)
        r=self.ask(s,'这10张里高于WHU全部样本平均变化像素的有多少张？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'n':8}])

    def test_non_null_filter_is_retained_when_region_changes(self):
        s=self.service(self.initial("region_id='whu' AND changed_px IS NOT NULL")+self.compare()+[
            contract(regions=['levir'],scopes=scopes('levir','reselect')),self.compare()[1]])
        self.first(s);self.ask(s,'这10张高于WHU全部均值的有多少张？')
        r=self.ask(s,'换成LEVIR，也取前10张，与LEVIR自己的均值比较。')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'n':1}])
        self.assertIn('IS NULL',r['sqlQuery'])

    def test_region_proof_uses_actual_rows_not_literals_in_negative_filter(self):
        initial=[contract(regions=[],columns=['sample_id','changed_px'],order=[dict(expression='changed_px',direction='DESC'),dict(expression='sample_id',direction='ASC')],limit=10),
                 "SELECT sample_id,changed_px FROM samples WHERE region_id<>'whu' ORDER BY changed_px DESC,sample_id ASC LIMIT 10"]
        value=scopes('levir')
        bad="SELECT COUNT(*) n FROM selected_samples WHERE region_id='whu' AND changed_px>(SELECT AVG(changed_px) FROM reference_samples)"
        s=self.service(initial+[contract(regions=['levir'],scopes=value),bad,bad]);self.ask(s,'按变化像素降序列出一批样本，只返回样本ID和变化像素。')
        r=self.ask(s,'这10张与LEVIR全部样本均值相比，有多少张更高？')
        self.assertEqual(r['status'],'failed',r);self.assertEqual(r['sqlRows'],[])

    def test_row_comparison_binds_the_actual_returned_subset(self):
        value=scopes(comparison={'output':'rows'})
        sql='SELECT sample_id,changed_px FROM selected_samples WHERE changed_px>(SELECT AVG(changed_px) FROM reference_samples) ORDER BY changed_px DESC,sample_id ASC'
        s=self.service(self.initial()+[contract(columns=['sample_id','changed_px'],scopes=value),sql,
            contract(regions=['whu'],columns=['n']),'SELECT COUNT(*) n FROM samples'])
        self.first(s);r=self.ask(s,'这10张里列出高于WHU全部样本平均变化像素的样本。')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(len(r['sqlRows']),8)
        self.assertEqual(r['statisticalSelectionCount'],10);self.assertEqual(r['population']['sample_count'],8)
        r=self.ask(s,'这些样本共有多少张？')
        self.assertEqual(r['status'],'completed',r);self.assertEqual(r['sqlRows'],[{'n':8}])


if __name__=='__main__':unittest.main()
