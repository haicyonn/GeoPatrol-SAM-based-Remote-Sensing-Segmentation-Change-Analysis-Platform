import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_sql import ModelSQL


class SystemInfoTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'data.json'
        self.path.write_text(json.dumps({'regions':[{'id':'whu','short':'WHU','items':[{'id':'A','changedPx':10}]}]}),encoding='utf-8')

    def service(self,outputs=()):
        values=iter(outputs);self.calls=[]
        def generate(prompt,**kw):
            self.calls.append(prompt)
            return {'text':next(values),'finishReason':'stop','inputTokens':100,'outputTokens':20}
        return ModelSQL(self.path,Path(self.temp.name)/'h.sqlite',generate=generate,
            model_id='Qwen/Qwen2.5-Coder-7B-Instruct',contract_mode=True)

    def ask(self,s,q):return s.run(dict(question=q,sessionKey='s'))

    def plan(self,columns):
        return json.dumps(dict(action='query',question='WHU数据查询',regions=['whu'],columns=columns,
            order=[],limit=None,offset=0,per_group=False,calculations=[],missing=None))

    def test_identity_question_uses_runtime_configuration_without_sql_or_generation(self):
        s=self.service()
        for q in ['你是什么模型','你用的是什么模型？','现在用的是哪个模型？','你是DeepSeek还是Qwen？','你是谁？']:
            with self.subTest(q=q):
                r=self.ask(s,q)
                self.assertEqual(r['status'],'system_info')
                self.assertEqual(r['llmCalls'],0);self.assertEqual(r['sqlQuery'],'')
                self.assertIn(s.model_id,r['answer']);self.assertEqual(r['dataSource'],'runtime_configuration')
        self.assertEqual(self.calls,[])

    def test_system_information_does_not_overwrite_population(self):
        s=self.service([self.plan(['n']),"SELECT COUNT(*) n FROM samples WHERE region_id='whu'",
            self.plan(['mean']),'SELECT AVG(changed_px) mean FROM samples'])
        self.assertEqual(self.ask(s,'WHU有多少张样本？')['status'],'completed')
        self.assertEqual(self.ask(s,'你是什么模型？')['status'],'system_info')
        r=self.ask(s,'这些样本的平均变化像素是多少？')
        self.assertEqual(r['sqlRows'],[{'mean':10.0}]);self.assertTrue(r['population']['inherited'])

    def test_raw_model_sql_is_distinct_from_program_bound_sql(self):
        raw='SELECT AVG(changed_px) mean FROM samples'
        s=self.service([self.plan(['n']),"SELECT COUNT(*) n FROM samples WHERE region_id='whu'",self.plan(['mean']),raw])
        self.ask(s,'WHU有多少张样本？');r=self.ask(s,'这些样本的平均变化像素是多少？')
        self.assertEqual(r.get('modelGeneratedSQL'),raw)
        self.assertIn('sample_id IN',r['sqlQuery'])
        self.assertEqual(r['programChanges'],['bound_population'])
        self.assertEqual(r['attempts'][-1]['modelGeneratedSQL'],raw)

    def test_rejected_sql_keeps_the_models_actual_output_for_inspection(self):
        wrong="SELECT COUNT(*) n FROM samples WHERE region_id='wrong'"
        s=self.service([self.plan(['n']),wrong,wrong])
        r=self.ask(s,'WHU有多少张样本？')
        self.assertEqual(r['status'],'failed')
        rejected=[a for a in r['attempts'] if a['stage']=='sql']
        self.assertEqual([a.get('modelGeneratedSQL') for a in rejected],[wrong,wrong])

    def test_data_question_mentions_model_does_not_become_system_info(self):
        s=self.service([self.plan(['n']),"SELECT COUNT(*) n FROM samples WHERE region_id='whu' AND changed_px>0"])
        r=self.ask(s,'模型在WHU检出变化像素大于0的样本有多少张？')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['llmCalls'],2)

if __name__=='__main__':unittest.main()
