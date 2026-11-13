import importlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
api=importlib.import_module('serve_change_agent_api')


class ReviewApiTests(unittest.TestCase):
    def setUp(self):
        self.old=getattr(api,'ARGS',None)
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.restore)
        path=Path(self.temp.name)/'data.json'
        path.write_text(json.dumps({'regions':[{'id':'levir','short':'LEVIR','items':[
            {'id':'A','changedPx':300,'objectCount':2,'reviewLevel':'high','reviewReasonCodes':['large_hole_ratio']},
            {'id':'B','changedPx':200,'objectCount':1,'reviewLevel':'medium','reviewReasonCodes':['many_small_fragments']},
            {'id':'C','changedPx':100,'objectCount':1,'reviewLevel':'low'}]}]}),encoding='utf-8')
        api.ARGS=SimpleNamespace(no_auth=True,host='127.0.0.1',static_sql=True,
            mysql_host=None,mysql_user=None,use_deepseek=False,use_qwen=False,
            enable_agent_workflow=True,web_data=str(path),knowledge_dir=str(ROOT/'knowledge'),
            agent_checkpoint=str(Path(self.temp.name)/'agent.sqlite'),dense_retrieval=False,rerank=False,
            model_id='none',max_new_tokens=256,enable_rule_answer=True,enable_local_intent_corpus=False)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),api.Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)

    def restore(self):
        api.close_agent_workflows()
        if hasattr(api,'close_review_tasks'): api.close_review_tasks()
        api.ARGS=self.old

    def post(self,path,body,session='test'):
        body={**body,'sessionKey':session}
        req=urllib.request.Request(f'http://127.0.0.1:{self.server.server_port}{path}',
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req) as r:return r.status,json.load(r)

    def wait(self,job):
        for _ in range(100):
            _,data=self.post('/api/review_tasks',{'action':'status','jobId':job['jobId']})
            if data['state'] not in {'queued','running'}:return data['result']
            time.sleep(.01)
        self.fail('job not completed')

    def test_chat_submits_review_job_and_polling_returns_real_rows(self):
        status,job=self.post('/api/chat',{'question':'给我2张复核，兼顾碎片和空洞','region':{'id':'levir'}})
        self.assertEqual(status,202)
        result=self.wait(job)
        self.assertEqual([r['sample_id'] for r in result['evidenceSamples']],['A','B'])
        self.assertTrue(result['reviewPlan']['hardSatisfied'])

    def test_a_different_session_cannot_poll_the_job(self):
        _,job=self.post('/api/chat',{'question':'给我2张复核','region':{'id':'levir'}})
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.post('/api/review_tasks',{'action':'status','jobId':job['jobId']},session='other')
        self.assertEqual(e.exception.code,404)

    def test_ordinal_followup_is_bound_to_review_plan(self):
        _,job=self.post('/api/chat',{'question':'给我2张复核','region':{'id':'levir'}})
        p=self.wait(job)['reviewPlan']
        _,data=self.post('/api/chat',{'question':'第二张对象数量是多少？','region':{'id':'levir'},'reviewTaskId':p['taskId'],'reviewPlanVersion':p['version']})
        self.assertEqual(data['evidenceSamples'][0]['sample_id'],'B')


if __name__=='__main__':unittest.main()
