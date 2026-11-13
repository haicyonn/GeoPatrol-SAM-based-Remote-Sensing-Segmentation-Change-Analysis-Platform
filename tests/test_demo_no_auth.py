import importlib
import inspect
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
api = importlib.import_module('serve_change_agent_api')
demo = importlib.import_module('start_local_demo')


class DemoNoAuthTests(unittest.TestCase):
    def setUp(self):
        self.previous_args = getattr(api, 'ARGS', None)
        self.addCleanup(self.restore_args)
        api.ARGS = SimpleNamespace(no_auth=True, host='127.0.0.1', static_sql=True,
                                  mysql_host=None, mysql_user=None)

    def restore_args(self):
        if self.previous_args is not None:
            api.ARGS = self.previous_args

    def test_loopback_static_demo_accepts_an_empty_session(self):
        response = api.handle_auth_session('')
        self.assertTrue(response['loggedIn'])
        self.assertEqual(response['authMode'], 'local_demo')
        self.assertFalse(response['user']['isAdmin'])

    def test_real_database_configuration_does_not_disable_auth(self):
        api.ARGS.mysql_host, api.ARGS.mysql_user = 'localhost', 'test'
        self.assertFalse(api.handle_auth_session('')['loggedIn'])

    def test_non_loopback_host_does_not_disable_auth(self):
        api.ARGS.host = '0.0.0.0'
        self.assertFalse(api.handle_auth_session('')['loggedIn'])

    def test_non_static_mode_does_not_disable_auth(self):
        api.ARGS.static_sql = False
        self.assertFalse(api.handle_auth_session('')['loggedIn'])

    def test_proxy_injects_no_login_configuration_only_when_enabled(self):
        self.assertIn('no_auth', inspect.signature(demo.make_web_handler).parameters)
        handler = demo.make_web_handler('http://127.0.0.1:9999', ROOT / 'web', no_auth=True)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        source = urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/auth.js').read().decode()
        self.assertIn('window.__TRUST_AUTH_DISABLED__ = true', source)

    def test_backend_chat_accepts_no_token_in_local_demo(self):
        api.ARGS = SimpleNamespace(**vars(api.ARGS), use_deepseek=False, use_qwen=False,
            web_data=str(ROOT / 'web/app_data.json'), knowledge_dir=str(ROOT / 'knowledge'),
            enable_rule_answer=True, enable_local_intent_corpus=False, model_id='none',
            max_new_tokens=256, prompt_version='test')
        server = ThreadingHTTPServer(('127.0.0.1', 0), api.Handler)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/chat',
            data=json.dumps({'question': '什么是连通域？', 'transientSession': True,
                             'sessionKey': self.id()}).encode(), headers={'Content-Type': 'application/json'})
        response = json.load(urllib.request.urlopen(request))
        self.assertEqual(response['answerScope'], 'knowledge')
        self.assertTrue(response['citations'])


if __name__ == '__main__':
    unittest.main()
