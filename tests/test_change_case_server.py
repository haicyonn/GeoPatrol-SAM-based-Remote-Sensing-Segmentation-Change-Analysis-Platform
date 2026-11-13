from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
import json
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from change_case_server import make_handler, RequestError


class StubApp:
    def __init__(self):self.fail=None
    def status(self):return {'ready':True}
    def cases(self,offset,limit):return {'cases':[{'case_id':'val:q.png'}]}
    def image(self,cid,side):
        if cid!='val:q.png' or side not in ('A','B'):raise RequestError(404,'Unknown image')
        return b'fake PNG'
    def retrieve(self,payload):
        if payload.get('case_id')!='val:q.png':raise RequestError(404,'Unknown query')
        return {'references':[]}
    def answer(self,payload):
        if self.fail=='busy':raise RequestError(409,'Model busy; retry later')
        if self.fail=='model':raise RuntimeError('/private/model/path SECRET')
        return {'answer':'No visible change','references':[]}


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.app=StubApp();self.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.app))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.server.server_port)
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join()
    def request(self,path,body=None,headers=None):
        headers=headers or {}
        data=None if body is None else json.dumps(body).encode()
        if data is not None:headers.setdefault('Content-Type','application/json')
        try:
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(urllib.request.Request(self.url+path,data=data,headers=headers)) as r:
                return r.status,r.read()
        except urllib.error.HTTPError as e:return e.code,e.read()
    def test_unknown_paths_and_images_cannot_read_disk(self):
        for path in ('/../scoring_only.jsonl','/api/images?case_id=../../secret&side=A','/api/images?case_id=val:q.png&side=..'):
            self.assertEqual(self.request(path)[0],404)
        self.assertEqual(self.request('/api/cases')[0],200)
        self.assertNotIn(b'captions',self.request('/api/cases')[1])
    def test_request_bounds_and_foreign_origin(self):
        self.assertEqual(self.request('/api/answer',{'question':'x'*9000})[0],413)
        self.assertEqual(self.request('/api/answer',[],{})[0],400)
        self.assertEqual(self.request('/api/answer',{}, {'Origin':'https://evil.example'})[0],403)
        self.assertEqual(self.request('/api/answer',{}, {'Host':'evil.example'})[0],403)
        self.assertEqual(self.request('/api/cases?limit=999999')[0],400)
    def test_busy_and_model_error_are_explicit_not_leaked(self):
        self.app.fail='busy';self.assertEqual(self.request('/api/answer',{})[0],409)
        self.app.fail='model';status,body=self.request('/api/answer',{})
        self.assertEqual(status,503);self.assertNotIn(b'SECRET',body)
        self.app.fail=None;self.assertEqual(self.request('/api/answer',{})[0],200)

if __name__=='__main__':unittest.main()
