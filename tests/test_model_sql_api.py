import importlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import sqlite3
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
api = importlib.import_module('serve_change_agent_api')


class ModelSQLApiTests(unittest.TestCase):
    def model_reply(self,sql,columns):
        def reply(prompt,**kwargs):
            if prompt.startswith('You interpret'):
                value={'action':'query','question':'查询当前问题','regions':['levir'],'columns':columns,
                    'order':[],'limit':None,'offset':0,'per_group':False,'calculations':[],'missing':None}
                return {'text':json.dumps(value),'finishReason':'stop'}
            return {'text':sql,'finishReason':'stop'}
        return reply

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        old = getattr(api, 'ARGS', None); self.addCleanup(lambda: setattr(api, 'ARGS', old))
        self.addCleanup(lambda: getattr(api, 'MODEL_SQL_SERVICES', {}).clear())
        self.addCleanup(lambda: getattr(api, 'CONVERSATION_SERVICES', {}).clear())
        self.addCleanup(api.close_review_tasks)
        source = Path(self.temp.name) / 'data.json'
        from PIL import Image
        (Path(self.temp.name)/'assets').mkdir()
        Image.new('L',(10,10),255).save(Path(self.temp.name)/'assets/a.png')
        source.write_text(json.dumps({'regions': [{'id': 'levir', 'short': 'LEVIR', 'items': [
            {'id': 'A', 'changedPx': 100, 'objectCount': 2,'assets':{'mask':'assets/a.png'}}]}]}), encoding='utf-8')
        api.ARGS = SimpleNamespace(enable_model_sql=True, use_deepseek=True, use_qwen=False,
            no_auth=True, host='127.0.0.1', static_sql=True, mysql_host=None, mysql_user=None,
            web_data=str(source), model_id='Qwen/test-model', enable_agent_workflow=False,
            agent_checkpoint=str(Path(self.temp.name) / 'agent.sqlite'))
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), api.Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close); self.addCleanup(self.server.shutdown)

    def post(self, path, body):
        req = urllib.request.Request(f'http://127.0.0.1:{self.server.server_port}{path}',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as r: return r.status, json.load(r)

    def test_new_endpoint_uses_model_and_returns_actual_query_results(self):
        with patch.object(api, 'deepseek_generate_raw', side_effect=self.model_reply('SELECT COUNT(*) n FROM samples',['n'])) as model:
            status, data = self.post('/api/nl_sql', {'question': '多少条', 'sessionKey': 'one'})
        self.assertEqual(status, 200); self.assertEqual(data['sqlRows'], [{'n': 1}])
        self.assertEqual(model.call_count, 2); self.assertEqual(data['modelId'], 'Qwen/test-model')
        self.assertEqual(data['queryContract']['columns'],['n'])

    def test_unified_endpoint_replies_to_chat_without_sample_queries(self):
        from model_sql import PublicQueryData
        with patch.object(api,'deepseek_generate_raw',return_value={'text':'你好，我是本地助手。','finishReason':'stop'}), \
             patch.object(PublicQueryData,'execute',side_effect=AssertionError('Chat queried samples')):
            status,data=self.post('/api/assistant',dict(question='你好',sessionKey='chat'))
            _,history=self.post('/api/assistant',dict(action='history',sessionKey='chat'))
        self.assertEqual(status,200);self.assertEqual(data['status'],'chat_completed');self.assertFalse(data['usedSQL'])
        self.assertEqual(len(history['messages']),2)

    def test_unified_endpoint_preserves_query_authorization_and_reports_execution(self):
        with patch.object(api,'deepseek_generate_raw',side_effect=self.model_reply('SELECT COUNT(*) n FROM samples WHERE region_id=\'levir\'',['n'])):
            _,data=self.post('/api/assistant',dict(question='统计LEVIR样本数量',sessionKey='query'))
        self.assertEqual(data['status'],'completed',data);self.assertTrue(data['usedSQL'])
        self.assertEqual(data['sqlRows'],[dict(n=1)]);self.assertEqual(data['sqlExecutionCount'],1)

    def test_unified_endpoint_obeys_disabled_feature(self):
        api.ARGS.enable_model_sql=False
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post('/api/assistant',dict(question='你好',sessionKey='disabled'))
        self.assertEqual(error.exception.code,400)

    def test_model_identity_is_runtime_info_not_a_failed_sql_contract(self):
        with patch.object(api,'deepseek_generate_raw',side_effect=AssertionError('Identity must not invoke generation')):
            status,data=self.post('/api/nl_sql',{'question':'你是什么模型','sessionKey':'identity'})
        self.assertEqual(status,200);self.assertEqual(data['status'],'system_info')
        self.assertEqual(data['llmCalls'],0);self.assertEqual(data['sqlQuery'],'')
        self.assertIn('Qwen/test-model',data['answer'])

    def test_explicit_chat_mode_bypasses_legacy_keyword_templates(self):
        with patch.object(api, 'deepseek_generate_raw', side_effect=self.model_reply('SELECT SUM(object_count) total FROM samples',['total'])), \
             patch.object(api, 'handle_chat_legacy', side_effect=AssertionError('Legacy route used')):
            _, data = self.post('/api/chat', {'question': '一共有几个对象', 'sessionKey': 'one', 'queryMode': 'model_sql'})
        self.assertEqual(data['sqlRows'], [{'total': 2}])

    def test_existing_manual_confirmation_is_available_to_model_queries(self):
        version = hashlib.sha256(Path(api.ARGS.web_data).read_bytes()).hexdigest()
        db = sqlite3.connect(Path(self.temp.name) / 'review_tasks.sqlite')
        try:
            db.execute('CREATE TABLE review_events(source_version TEXT,sample_id TEXT)')
            db.execute('INSERT INTO review_events VALUES(?,?)', (version, 'A')); db.commit()
        finally: db.close()
        with patch.object(api, 'deepseek_generate_raw', side_effect=self.model_reply('SELECT COUNT(*) n FROM samples WHERE reviewed=1',['n'])):
            _,data = self.post('/api/nl_sql', {'question':'已复核多少个','sessionKey':'one'})
        self.assertEqual(data['sqlRows'], [{'n':1}])

    def test_scope_guard_returns_actionable_clarification_without_generation(self):
        p=Path(api.ARGS.web_data);d=json.loads(p.read_text(encoding='utf-8'))
        d['regions'].append({'id':'whu','short':'WHU','items':[]});p.write_text(json.dumps(d),encoding='utf-8')
        with patch.object(api,'deepseek_generate_raw',side_effect=AssertionError('Model must not run')):
            _,r=self.post('/api/nl_sql',{'question':'比较LEVIR和WHU数量','scopeRegion':'whu','sessionKey':'guard'})
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],0)

    def test_disabled_feature_cannot_generate_or_execute_queries(self):
        api.ARGS.enable_model_sql = False
        with patch.object(api, 'deepseek_generate_raw', side_effect=AssertionError('Disabled model called')):
            with self.assertRaises(urllib.error.HTTPError) as exc:
                self.post('/api/nl_sql', {'question': '多少条', 'sessionKey': 'one'})
        self.assertEqual(exc.exception.code, 400)

    def test_query_review_http_roundtrip_and_confirmed_count(self):
        with patch.object(api,'deepseek_generate_raw',side_effect=self.model_reply('SELECT sample_id FROM samples',['sample_id'])):
            _,query=self.post('/api/nl_sql',{'question':'所有样本ID','sessionKey':'bridge'})
        _,plan=self.post('/api/review_tasks',{'action':'from_query','sessionKey':'bridge',
            'dataVersion':query['dataVersion'],'sampleIds':['FORGED'],'sql':'SELECT secret FROM private'})
        self.assertEqual([x['sample_id'] for x in plan['evidenceSamples']],['A'])
        row=plan['evidenceSamples'][0]
        self.post('/api/review_tasks',{'action':'confirm','sessionKey':'bridge',
            'reviewTaskId':plan['reviewPlan']['taskId'],'reviewPlanVersion':1,
            'sampleId':'A','factsHash':row['facts_hash'],'note':'测试夹具人工已核对','requestId':'r1'})
        with patch.object(api,'deepseek_generate_raw',side_effect=self.model_reply('SELECT COUNT(*) n FROM samples WHERE reviewed=1',['n'])):
            _,after=self.post('/api/nl_sql',{'question':'已复核多少张','sessionKey':'bridge'})
        self.assertEqual(after['sqlRows'],[{'n':1}]);self.assertNotEqual(after['dataVersion'],query['dataVersion'])


if __name__ == '__main__': unittest.main()
