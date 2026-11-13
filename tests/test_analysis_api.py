import json
import unittest
from unittest.mock import patch
import urllib.error
import test_model_sql_api as fixture
api = fixture.api


class AnalysisApiTests(unittest.TestCase):
    setUp = fixture.ModelSQLApiTests.setUp
    post = fixture.ModelSQLApiTests.post

    def tearDown(self):
        getattr(api,'ANALYSIS_TASK_SERVICES',{}).clear()

    def test_http_plan_execute_without_login_and_actual_query(self):
        plan={'title':'数量概况','clarification':None,'steps':[{'title':'样本数量','question':'LEVIR样本数量',
            'sql':'SELECT COUNT(*) AS n FROM samples'}]}
        report={'findings':[{'text':'样本数量为1。','evidence':['S1:R1']}],'nextQuestions':[]}
        with patch.object(api,'deepseek_generate_raw',side_effect=[{'text':json.dumps(plan),'finishReason':'stop'},
                {'text':json.dumps(report),'finishReason':'stop'}]):
            _,p=self.post('/api/analysis',dict(action='plan',goal='分析LEVIR的样本数量',sessionKey='analysis'))
            _,r=self.post('/api/analysis',dict(action='execute',planId=p['planId'],sessionKey='analysis'))
        self.assertEqual(r['steps'][0]['rows'],[{'n':1}])
        self.assertEqual(r['report']['status'],'generated')

    def test_http_cannot_forge_authenticated_owner(self):
        calls=[]
        def handle(payload):
            calls.append(payload)
            return {'status':'planned'}
        if not hasattr(api,'handle_analysis_task'):self.fail('Missing authenticated analysis handler')
        with patch.object(api,'handle_analysis_task',side_effect=handle):
            self.post('/api/analysis',dict(action='plan',goal='分析',sessionKey='a',_authenticatedUser='another-person'))
        self.assertNotEqual(calls[0]['_authenticatedUser'],'another-person')


if __name__=='__main__':unittest.main()
