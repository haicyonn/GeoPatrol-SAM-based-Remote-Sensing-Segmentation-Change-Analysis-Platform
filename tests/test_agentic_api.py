import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import serve_change_agent_api as api


@unittest.skipUnless(importlib.util.find_spec('langgraph'), 'optional agent runtime not installed')
class AgenticApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(lambda: getattr(api, 'close_agent_workflows', lambda: None)())
        api.ARGS = SimpleNamespace(use_deepseek=False, use_qwen=False, static_sql=True,
            web_data=str(ROOT / 'web/app_data.json'), knowledge_dir=str(ROOT / 'knowledge'),
            mysql_host=None, mysql_user=None, enable_rule_answer=True,
            enable_local_intent_corpus=False, model_id='none', max_new_tokens=256,
            prompt_version='test', enable_agent_workflow=True,
            dense_retrieval=False, rerank=False,
            agent_checkpoint=str(Path(self.temp.name) / 'agent.sqlite'),
            retrieval_cache=str(Path(self.temp.name) / 'model-cache'))

    def payload(self, q):
        return {'question': q, 'region': {'id': 'whu'}, 'transientSession': True,
                'sessionKey': self.id(), '_authenticatedUser': 'test-user'}

    def test_existing_chat_endpoint_uses_opted_in_graph(self):
        response = api.handle_chat(self.payload('哪些要优先看？'))
        self.assertEqual(response.get('agentRuntime'), 'langgraph')
        self.assertTrue(response['needsClarification'])

    def test_graph_drives_actual_snapshot_query(self):
        response = api.handle_chat(self.payload('当前区域变化像素前3名'))
        self.assertEqual(response.get('agentRuntime'), 'langgraph')
        self.assertEqual(len(response['sqlRows']), 3)
        self.assertEqual(response['llmCalls'], 0)
        self.assertEqual(response['dataSource'], 'static_snapshot')

    def test_hybrid_factory_configuration_is_explicit(self):
        api.ARGS.dense_retrieval = True
        index = api.inspection_knowledge_index()
        self.assertTrue(index.status().get('configured', {}).get('dense'))


if __name__ == '__main__':
    unittest.main()
