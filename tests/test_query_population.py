import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from model_sql import ModelSQL


def plan(question='查询', **changes):
    value = dict(action='query', question=question, regions=[], columns=['n'],
                 order=[], limit=None, offset=0, per_group=False, calculations=[], missing=None)
    value.update(changes)
    return json.dumps(value, ensure_ascii=False)


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'data.json'
        def sample(sid, pixels, flagged=True):
            return dict(id=sid, changedPx=pixels, objectCount=2,
                        reviewReasonCodes=['large_hole_ratio', 'many_small_fragments'] if flagged else [])
        self.source.write_text(json.dumps({'regions': [
            {'id': 'levir', 'short': 'LEVIR', 'items': [sample('A', 10), sample('B', 30), sample('C', 100, False)]},
            {'id': 'whu', 'short': 'WHU', 'items': [sample('D', 500)]}]}), encoding='utf-8')
        self.prompts = []

    def service(self, outputs):
        iterator = iter(outputs)
        def generate(prompt, **kw):
            self.prompts.append(prompt)
            return {'text': next(iterator), 'finishReason': 'stop'}
        return ModelSQL(self.source, Path(self.temp.name) / 'history.sqlite',
                        generate=generate, model_id='test', contract_mode=True)

    def ask(self, service, question, **kw):
        return service.run(dict(question=question, sessionKey='s', **kw))

    def count_outputs(self, empty=False):
        condition = "region_id='levir' AND has_holes=1 AND has_fragments=1" if not empty else "region_id='levir' AND changed_px>1000"
        return [plan('LEVIR空洞和碎片的数量', regions=['levir']),
                'SELECT COUNT(*) n FROM samples WHERE ' + condition]

    def test_filtered_count_followup_binds_population_even_when_model_drops_where(self):
        s = self.service(self.count_outputs() + [plan('平均变化像素', columns=['mean']),
                         'SELECT ROUND(AVG(changed_px),2) mean FROM samples'])
        self.assertEqual(self.ask(s, 'LEVIR中同时有空洞和碎片标记的样本共有多少张？')['sqlRows'], [{'n': 2}])
        r = self.ask(s, '这些样本的平均变化像素是多少？保留两位小数，只返回均值。')
        self.assertEqual(r['status'], 'completed')
        self.assertEqual(r['sqlRows'], [{'mean': 20.0}])
        self.assertTrue(r['population']['inherited'])
        self.assertIn('has_holes', r['sqlQuery'])
        self.assertIn('has_fragments', r['sqlQuery'])

    def test_explicit_rounding_is_enforced_independently_of_model_contract(self):
        s = self.service(self.count_outputs() + [plan(columns=['mean']),
                         'SELECT AVG(changed_px) mean FROM samples',
                         'SELECT ROUND(AVG(changed_px),2) mean FROM samples'])
        self.ask(s, 'LEVIR中同时有空洞和碎片标记的样本共有多少张？')
        r = self.ask(s, '这些样本平均变化像素是多少，保留两位小数？')
        self.assertEqual(r['status'], 'completed')
        self.assertIn('ROUND', r['sqlQuery'])
        self.assertTrue(any(a['status'] == 'rejected' for a in r['attempts']))

    def test_top_n_followup_uses_displayed_ids_not_all_filtered_rows(self):
        s = self.service([plan(regions=['levir'], columns=['sample_id'], order=[dict(expression='changed_px', direction='DESC')], limit=2),
            "SELECT sample_id FROM samples WHERE region_id='levir' ORDER BY changed_px DESC LIMIT 2",
            plan(columns=['total']), 'SELECT SUM(changed_px) total FROM samples'])
        self.ask(s, 'LEVIR变化像素最多的前2张，只返回样本ID')
        r = self.ask(s, '这两张样本总变化像素是多少？')
        self.assertEqual(r['sqlRows'], [{'total': 130.0}])

    def test_empty_population_remains_empty_for_followup(self):
        s = self.service(self.count_outputs(empty=True) + [plan(columns=['mean']), 'SELECT AVG(changed_px) mean FROM samples'])
        self.ask(s, 'LEVIR变化像素大于1000的样本多少张')
        self.assertEqual(self.ask(s, '这些样本的平均变化像素是多少')['sqlRows'], [{'mean': None}])

    def test_missing_population_clarifies_without_generation(self):
        s = self.service([])
        r = self.ask(s, '这些样本平均变化像素是多少')
        self.assertEqual(r['status'], 'needs_clarification')
        self.assertEqual(r['llmCalls'], 0)

    def test_population_isolated_by_identity_and_service_restart(self):
        s = self.service(self.count_outputs()); self.ask(s, 'LEVIR中同时有空洞和碎片标记的样本共有多少张？')
        other = self.service([])
        self.assertEqual(self.ask(other, '这些样本平均变化像素是多少', _authenticatedUser='other')['status'], 'needs_clarification')
        restored = self.service([plan(columns=['mean']), 'SELECT AVG(changed_px) mean FROM samples'])
        self.assertEqual(self.ask(restored, '这些样本平均变化像素是多少')['sqlRows'], [{'mean': 20.0}])

    def test_fresh_question_does_not_inherit_previous_population(self):
        s = self.service(self.count_outputs() + [plan(), 'SELECT COUNT(*) n FROM samples'])
        self.ask(s, 'LEVIR中同时有空洞和碎片标记的样本共有多少张？')
        self.assertEqual(self.ask(s, '全部区域样本总数是多少')['sqlRows'], [{'n': 4}])

    def test_cte_shadow_cannot_bypass_inherited_population(self):
        bad = 'WITH samples AS (SELECT sample_count FROM regions) SELECT SUM(sample_count) n FROM samples'
        s = self.service(self.count_outputs() + [plan(), bad, bad])
        self.ask(s, 'LEVIR中同时有空洞和碎片标记的样本共有多少张？')
        r = self.ask(s, '这些样本总数是多少')
        self.assertEqual(r['status'], 'failed')
        self.assertEqual(r['sqlRows'], [])

    def test_conditional_aggregates_do_not_claim_the_outer_filter_is_the_cohort(self):
        for expression in ['COUNT(CASE WHEN has_holes=1 AND has_fragments=1 THEN 1 END)',
                           'COUNT(*) FILTER(WHERE has_holes=1 AND has_fragments=1)',
                           'COUNT(DISTINCT object_count)']:
            with self.subTest(expression=expression):
                s=self.service([plan(regions=['levir']),"SELECT "+expression+" n FROM samples WHERE region_id='levir'"])
                self.assertEqual(self.ask(s,'LEVIR条件统计')['status'],'completed')
                r=self.ask(s,'这些样本平均变化像素是多少？')
                self.assertEqual(r['status'],'needs_clarification')
                self.assertEqual(r['llmCalls'],0)

    def test_population_expansion_clarifies_without_silently_ignoring_added_region(self):
        s=self.service(self.count_outputs())
        self.ask(s,'LEVIR中同时有空洞和碎片标记的样本共有多少张？')
        r=self.ask(s,'这些样本再加上WHU的样本一起计算平均变化像素，保留两位小数')
        self.assertEqual(r['status'],'needs_clarification');self.assertEqual(r['llmCalls'],0)

    def test_mean_expression_cannot_add_arbitrary_offset(self):
        bad='SELECT ROUND(AVG(changed_px),2)+1000 mean FROM samples'
        s=self.service(self.count_outputs()+[plan(columns=['mean']),bad,bad])
        self.ask(s,'LEVIR中同时有空洞和碎片标记的样本共有多少张？')
        r=self.ask(s,'这些样本平均变化像素是多少，保留两位小数')
        self.assertEqual(r['status'],'failed');self.assertEqual(r['sqlRows'],[])

    def test_share_within_bound_population_does_not_require_external_scope(self):
        s=self.service(self.count_outputs()+[plan(columns=['pct']),
            'SELECT 100.0*SUM(CASE WHEN has_holes=1 THEN 1 ELSE 0 END)/COUNT(*) pct FROM samples'])
        self.ask(s,'LEVIR中同时有空洞和碎片标记的样本共有多少张？')
        r=self.ask(s,'这些样本中带有空洞标记的占比是多少？')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['sqlRows'],[{'pct':100.0}])

    def test_repeated_filter_does_not_replace_cohort_origin_with_deictic_question(self):
        first='LEVIR中同时有空洞和碎片标记的样本共有多少张？'
        sql="SELECT AVG(changed_px) mean FROM samples WHERE region_id='levir' AND has_holes=1 AND has_fragments=1"
        s=self.service(self.count_outputs()+[plan(columns=['mean']),sql])
        self.ask(s,first)
        r=self.ask(s,'这些样本平均变化像素是多少')
        self.assertEqual(r['sqlRows'],[{'mean':20.0}])
        self.assertEqual(r['population']['origin_question'],first)


if __name__ == '__main__': unittest.main()
