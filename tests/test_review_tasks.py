import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from review_tasks import ReviewTasks, ReviewError


class ReviewTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'data.json'
        items = []
        for i in range(12):
            code = ['large_hole_ratio'] if i in (1, 4, 7) else ['many_small_fragments'] if i in (2, 5) else []
            items.append({'id': f'S{i:02}', 'changedPx': 1000-i*10, 'changedRatio': .2,
                          'objectCount': i, 'reviewLevel': 'high', 'reviewReasonCodes': code,
                          'maskQuality': {'hole_ratio': .3 if code == ['large_hole_ratio'] else 0}})
        self.source.write_text(json.dumps({'regions': [{'id': 'levir', 'short': 'LEVIR', 'items': items}]}), encoding='utf-8')
        self.services = []
        self.service = self.make_service()
        self.addCleanup(lambda: [s.close() for s in self.services])

    def make_service(self, generate=None):
        s = ReviewTasks(self.source, self.root / 'review.sqlite', generate=generate, retrieve=lambda q: [])
        self.services.append(s)
        return s

    def payload(self, question, **extra):
        return {'question': question, 'region': {'id': 'levir'}, 'sessionKey': 'one',
                '_authenticatedUser': 'alice', **extra}

    def ask(self, q, **extra):
        payload = self.payload(q, **extra)
        job = self.service.submit(payload)
        owner = self.service.owner(payload)
        for _ in range(300):
            state = self.service.get(job['jobId'], owner)
            if state['state'] not in {'queued', 'running'}:
                self.assertNotEqual(state['state'], 'failed', state)
                return state['result']
            time.sleep(.005)
        self.fail('job did not finish')

    def test_task_is_real_and_has_checkable_coverage(self):
        result = self.ask('给我3张复核，兼顾碎片和空洞，排除已复核')
        self.assertEqual(len(result['evidenceSamples']), 3)
        self.assertTrue(result['reviewPlan']['hardSatisfied'])
        self.assertGreater(result['reviewPlan']['coverage']['holes'], 0)
        self.assertEqual(result['llmCalls'], 0)
        self.assertEqual(result['plannerStatus'], 'rules_only')

    def test_model_interprets_colloquial_preferences_and_is_measured(self):
        self.service = self.make_service(generate=lambda *a, **kw: {'text': json.dumps({'coverage': ['fragmentation', 'holes']}), 'finishReason': 'stop', 'outputTokens': 18})
        result = self.ask('选3张复核，撒芝麻和蜂窝那种都照顾到')
        self.assertEqual(result['llmCalls'], 1)
        self.assertEqual(result['plannerStatus'], 'model')
        self.assertEqual(set(result['reviewPlan']['conditions']['coverage']), {'fragmentation', 'holes'})

    def test_impossible_explicit_quota_is_reported_without_wasting_a_model_call(self):
        def unexpected(*a, **kw): self.fail('Known shortage must be checked before generation')
        self.service = self.make_service(generate=unexpected)
        result = self.ask('选10张复核，空洞至少9张')
        self.assertFalse(result['reviewPlan']['hardSatisfied'])
        self.assertEqual(result['llmCalls'], 0)
        self.assertEqual(result['plannerStatus'], 'preflight_shortage')

    def test_saved_confirmation_is_visible_when_the_task_is_fetched_again(self):
        first = self.ask('给我3张复核')
        p = first['reviewPlan']; sample = first['evidenceSamples'][0]
        data = self.payload('', reviewTaskId=p['taskId'], reviewPlanVersion=p['version'],
            sampleId=sample['sample_id'], factsHash=sample['facts_hash'],
            note='测试库中的人工结论', requestId='refresh')
        owner = self.service.owner(data)
        self.service.confirm(data, owner)
        with self.service.db() as db:
            job = db.execute('SELECT job_id FROM review_jobs ORDER BY created DESC LIMIT 1').fetchone()[0]
        fetched = self.service.get(job, owner)['result']
        self.assertIn(sample['sample_id'], fetched['reviewConfirmedIds'])

    def test_invalid_model_output_is_repaired_once_and_not_labeled_success(self):
        calls = []
        def generate(*a, **kw):
            calls.append(1)
            return {'text': 'not json', 'finishReason': 'length'}
        self.service = self.make_service(generate=generate)
        result = self.ask('选3张复核，蜂窝那种')
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['plannerStatus'], 'model_failed')
        self.assertFalse(result['reviewPlan']['hardSatisfied'])

    def test_revision_is_versioned_and_retains_the_other_rows(self):
        first = self.ask('给我3张复核，排除已复核')
        p = first['reviewPlan']
        result = self.ask('第三张移出，再补一张空洞类的', reviewTaskId=p['taskId'], reviewPlanVersion=p['version'])
        self.assertEqual(result['reviewPlan']['version'], 2)
        self.assertEqual([r['sample_id'] for r in result['evidenceSamples'][:2]], [r['sample_id'] for r in first['evidenceSamples'][:2]])
        self.assertNotEqual(result['evidenceSamples'][2]['sample_id'], first['evidenceSamples'][2]['sample_id'])
        self.assertEqual(self.service.reviewed_ids(), set())

    def test_ordinal_question_uses_the_review_list(self):
        first = self.ask('给我3张复核')
        p = first['reviewPlan']
        result = self.service.reference(self.payload('第二张对象数量是多少？', reviewTaskId=p['taskId'], reviewPlanVersion=p['version']))
        self.assertEqual(result['evidenceSamples'][0]['sample_id'], first['evidenceSamples'][1]['sample_id'])

    def test_manual_confirmation_is_idempotent_persists_and_excludes(self):
        first = self.ask('给我3张复核')
        p = first['reviewPlan']; sample = first['evidenceSamples'][0]
        data = self.payload('', reviewTaskId=p['taskId'], reviewPlanVersion=p['version'],
            sampleId=sample['sample_id'], factsHash=sample['facts_hash'], note='已对照双时相影像，保留检测结果。', requestId='same')
        owner = self.service.owner(data)
        once = self.service.confirm(data, owner)
        twice = self.service.confirm(data, owner)
        self.assertEqual(once, twice)
        self.service.close(); self.services.remove(self.service)
        self.service = self.make_service()
        result = self.ask('再给我3张复核，排除已复核')
        self.assertNotIn(sample['sample_id'], [r['sample_id'] for r in result['evidenceSamples']])

    def test_missing_note_is_not_an_automatic_human_confirmation(self):
        first = self.ask('给我3张复核'); p = first['reviewPlan']; s = first['evidenceSamples'][0]
        data = self.payload('', reviewTaskId=p['taskId'], reviewPlanVersion=1, sampleId=s['sample_id'], factsHash=s['facts_hash'], requestId='empty')
        with self.assertRaises(ReviewError): self.service.confirm(data, self.service.owner(data))
        self.assertEqual(self.service.reviewed_ids(), set())

    def test_old_version_and_foreign_session_cannot_mutate_or_read_task(self):
        first = self.ask('给我3张复核'); p = first['reviewPlan']
        self.ask('第三张移出，再补一张', reviewTaskId=p['taskId'], reviewPlanVersion=1)
        with self.assertRaises(ReviewError):
            self.service.submit(self.payload('第二张换掉', reviewTaskId=p['taskId'], reviewPlanVersion=1))
        foreign = self.payload('第二张换掉', reviewTaskId=p['taskId'], reviewPlanVersion=2, sessionKey='other')
        with self.assertRaises(ReviewError): self.service.submit(foreign)

    def test_queue_is_bounded_and_cancellation_does_not_create_a_plan(self):
        entered, release = threading.Event(), threading.Event()
        def generate(*a, **kw):
            entered.set(); release.wait(2)
            return {'text': '{}', 'finishReason': 'stop'}
        self.service = self.make_service(generate=generate)
        data=self.payload('给我3张复核'); owner=self.service.owner(data)
        job=self.service.submit(data); self.assertTrue(entered.wait(1))
        queued=self.service.submit(data)
        with self.assertRaises(ReviewError): self.service.submit(data)
        self.service.cancel(queued['jobId'], owner)
        release.set()
        time.sleep(.05)
        self.assertEqual(self.service.get(queued['jobId'],owner)['state'],'cancelled')

    def test_changed_source_rejects_old_fact_confirmation(self):
        first=self.ask('给我3张复核'); p=first['reviewPlan']; s=first['evidenceSamples'][0]
        data=json.loads(self.source.read_text()); data['regions'][0]['items'][0]['changedPx']=99999
        self.source.write_text(json.dumps(data),encoding='utf-8')
        request=self.payload('',reviewTaskId=p['taskId'],reviewPlanVersion=1,sampleId=s['sample_id'],factsHash=s['facts_hash'],note='已查看',requestId='stale')
        with self.assertRaises(ReviewError): self.service.confirm(request,self.service.owner(request))

    def test_late_cancel_after_commit_does_not_hide_an_already_saved_plan(self):
        original=self.service._execute
        def late(*args):
            result=original(*args)
            args[3].set()  # cancellation arrives after the final commit
            return result
        with patch.object(self.service,'_execute',side_effect=late):
            result=self.ask('给我3张复核')
        self.assertIsNotNone(result)
        self.assertTrue(result['reviewPlan']['hardSatisfied'])


if __name__ == '__main__': unittest.main()
