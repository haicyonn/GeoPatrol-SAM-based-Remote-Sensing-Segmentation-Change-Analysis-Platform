import importlib
import json
import sys
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
demo = importlib.import_module('start_local_demo')


class DemoProxyTests(unittest.TestCase):
    def test_browser_uses_same_origin_and_post_reaches_api(self):
        self.assertTrue(hasattr(demo, 'make_web_handler'), 'Missing same-origin demo API proxy')

        class Echo(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                body = json.dumps({'answer':payload['question'],'citations':[{'id':1}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.end_headers()
                self.wfile.write(body)

        upstream = ThreadingHTTPServer(('127.0.0.1',0),Echo)
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        threading.Thread(target=upstream.serve_forever,daemon=True).start()
        handler = demo.make_web_handler(f'http://127.0.0.1:{upstream.server_port}',Path(__file__).parent)
        server = ThreadingHTTPServer(('127.0.0.1',0),handler)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        base = f'http://127.0.0.1:{server.server_port}'
        config = json.load(urllib.request.urlopen(base+'/runtime_config.json'))
        self.assertEqual(config['qaApiUrl'],base+'/api/chat')
        req = urllib.request.Request(config['qaApiUrl'],data=json.dumps({'question':'连通域'}).encode(),headers={'Content-Type':'application/json'})
        data = json.load(urllib.request.urlopen(req))
        self.assertEqual(data['answer'],'连通域')
        self.assertEqual(data['citations'],[{'id':1}])
