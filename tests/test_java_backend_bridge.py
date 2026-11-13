import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import java_backend_bridge as adapter


class BackendBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.assets=self.root/'assets'; self.assets.mkdir()
        self.data={'regions':[{'id':'whu','short':'WHU','items':[
            {'id':'fresh-1','changedPx':4,'changedRatio':.25,'objectCount':1},
            {'id':'fresh-2','changedPx':8,'changedRatio':.5,'objectCount':2}]}]}
        self.bridge=adapter.Bridge(self.assets,self.root/'state',generate=None)

    def test_relative_and_symlink_escapes_cannot_be_read(self):
        (self.assets/'link.png').symlink_to(self.root/'secret.png')
        for name in ['../secret.png','/tmp/secret.png','link.png']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                adapter.safe_asset(self.assets,name)

    def test_registered_dataset_is_immutable(self):
        first=self.bridge.register_dataset('batch-one',self.data)
        self.assertEqual(first,self.bridge.register_dataset('batch-one',self.data))
        with self.assertRaises(ValueError):
            self.bridge.register_dataset('batch-one',{'regions':[]})

    def test_registration_rejects_nonfinite_and_duplicate_sample_ids(self):
        for bad in [float('nan'),-1]:
            data=json.loads(json.dumps(self.data)); data['regions'][0]['items'][0]['changedPx']=bad
            with self.assertRaises(ValueError):self.bridge.register_dataset('invalid',data)
        self.data['regions'][0]['items'][1]['id']='fresh-1'
        with self.assertRaises(ValueError):self.bridge.register_dataset('invalid',self.data)

    def test_new_database_rows_are_the_actual_sql_source_after_restart(self):
        self.bridge.register_dataset('batch-one',self.data)
        q='共有多少样本？只返回数量。'
        outputs=iter([json.dumps({'action':'query','question':q,'regions':[],
            'columns':['sample_count'],'order':[],'limit':None,'offset':0,
            'per_group':False,'calculations':['COUNT(*)'],'missing':None}),
            'SELECT COUNT(*) AS sample_count FROM samples'])
        restarted=adapter.Bridge(self.assets,self.root/'state',generate=lambda *a,**k:next(outputs))
        result=restarted.query('batch-one',{'sessionId':'session','userId':'member-a','question':q})
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual(result['sqlRows'],[{'sample_count':2}])
        self.assertIn('COUNT',result['sqlQuery'].upper())

    def test_unknown_dataset_and_client_file_paths_are_rejected(self):
        with self.assertRaises(ValueError):self.bridge.query('../file',{'sessionId':'x','question':'多少'})
        with self.assertRaises(ValueError):self.bridge.query('missing',{'sessionId':'x','question':'多少'})

    def test_internal_http_endpoint_requires_service_token(self):
        server=adapter.create_server(self.bridge,'test-service-token-with-at-least-32-chars',0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        request=urllib.request.Request('http://127.0.0.1:'+str(server.server_port)+'/internal/datasets',
            data=json.dumps({'datasetId':'batch-one','data':self.data}).encode(),headers={'Content-Type':'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as raised:urllib.request.urlopen(request)
        self.assertEqual(raised.exception.code,403)

    def test_internal_responses_explicitly_close_the_http10_connection(self):
        import http.client
        token='test-service-token-with-at-least-32-chars'
        server=adapter.create_server(self.bridge,token,0)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
        self.addCleanup(conn.close)
        conn.request('GET','/internal/health',headers={'X-Xunbian-Token':token,'Connection':'keep-alive'})
        response=conn.getresponse();response.read()
        self.assertEqual(response.getheader('Connection'),'close')

    def test_same_browser_session_id_has_distinct_user_query_history(self):
        q='共有多少样本？只返回数量。'
        plan=json.dumps({'action':'query','question':q,'regions':[], 'columns':['sample_count'],
            'order':[],'limit':None,'offset':0,'per_group':False,'calculations':['COUNT(*)'],'missing':None})
        outputs=iter([plan,'SELECT COUNT(*) AS sample_count FROM samples']*2)
        bridge=adapter.Bridge(self.assets,self.root/'state',generate=lambda *a,**k:next(outputs))
        bridge.register_dataset('shared-batch',self.data)
        for user in ('member-a','member-b'):
            result=bridge.query('shared-batch',{'sessionId':'same-client-session','userId':user,'question':q})
            self.assertEqual(result['status'],'completed',result)
        with sqlite3.connect(self.root/'state/shared-batch-assistant/model_sql.sqlite') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM query_history').fetchone()[0],2)

    def test_missing_trusted_user_identity_is_rejected(self):
        self.bridge.register_dataset('batch-one',self.data)
        self.bridge.generate=lambda *a,**k:'{}'
        with self.assertRaisesRegex(ValueError,'用户'):
            self.bridge.query('batch-one',{'sessionId':'same','question':'多少样本'})

    def test_greeting_uses_original_conversation_route_without_sample_query(self):
        self.bridge.generate=lambda *args,**kwargs: {'text':'你好！有什么可以帮你？','finishReason':'stop'}
        self.bridge.register_dataset('batch-one',self.data)
        result=self.bridge.query('batch-one',{'sessionId':'hello-session','userId':'member-a','question':'你好'})
        self.assertEqual(result['status'],'chat_completed',result)
        self.assertFalse(result['usedSQL']);self.assertEqual(result['sqlExecutionCount'],0)
        history=self.bridge.query('batch-one',{'sessionId':'hello-session','userId':'member-a','action':'history'})
        self.assertEqual(history['status'],'history');self.assertEqual(history['messages'][0]['content'],'你好')

    def test_visible_region_is_forwarded_to_original_assistant(self):
        self.bridge.register_dataset('batch-one',self.data)
        q='你好吗，这个区域有几个图片'
        plan=json.dumps(dict(action='query',question='WHU样本数量',regions=['whu'],columns=['sample_count'],order=[],limit=None,offset=0,per_group=False,calculations=['COUNT(*)'],missing=None))
        outputs=iter([plan,"SELECT COUNT(*) AS sample_count FROM samples WHERE region_id='whu'"])
        self.bridge.generate=lambda *a,**k:next(outputs)
        r=self.bridge.query('batch-one',dict(sessionId='context',userId='member-a',question=q,viewRegion='whu'))
        self.assertEqual(r['sqlRows'],[{'sample_count':2}],r)
        self.assertEqual(r['questionContext']['viewRegion'],'whu')


if __name__=='__main__':unittest.main()
