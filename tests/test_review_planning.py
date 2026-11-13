import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from review_planning import Snapshot, compile_request, select_batch, is_review_request

REGIONS = [{'id': 'levir', 'short': 'LEVIR', 'name': '建筑变化'},
           {'id': 'whu', 'short': 'WHU', 'name': '武汉变化'}]


def row(name, families=(), risk='high', px=100, objects=2):
    return {'sample_id': name, 'region_id': 'levir', 'families': list(families),
            'review_level': risk, 'changed_px': px, 'object_count': objects,
            'changed_ratio': .2, 'quality': {}, 'facts_hash': name}


class RequestTests(unittest.TestCase):
    def test_elided_second_bound_is_not_lost_when_anchoring_model_plan(self):
        p = compile_request('选1张复核，变化像素大于100且小于500', 'levir', REGIONS,
            model_plan={'filters': [{'field': 'changed_px', 'op': 'gt', 'value': 100},
                                    {'field': 'changed_px', 'op': 'lt', 'value': 500}]})
        result = select_batch([row('A', px=900), row('B', px=200)], p)
        self.assertEqual([r['sample_id'] for r in result['rows']], ['B'])
        self.assertTrue(result['hardSatisfied'])

    def test_explicit_threshold_replacement_preserves_other_bounds_and_fields(self):
        old = compile_request('选1张复核，变化像素大于1000且小于2000，对象数量小于5', 'levir', REGIONS)
        p = compile_request('其他条件保持，改为变化像素大于500', 'levir', REGIONS,
                            previous={'plan': old, 'rows': []})
        self.assertNotIn({'field': 'changed_px', 'op': 'gt', 'value': 1000}, p['filters'])
        self.assertIn({'field': 'changed_px', 'op': 'lt', 'value': 2000}, p['filters'])
        result = select_batch([row('A', px=700)], p)
        self.assertFalse(p['issues'])
        self.assertTrue(result['hardSatisfied'])

    def test_short_threshold_revision_is_not_mistaken_for_a_region(self):
        old = compile_request('选1张复核，变化像素大于1000', 'levir', REGIONS)
        p = compile_request('改为变化像素大于500', 'levir', REGIONS,
                            previous={'plan': old, 'rows': []})
        self.assertFalse(p['issues'])

    def test_only_family_is_a_filter_rather_than_a_soft_preference(self):
        p = compile_request('选3张复核，只选空洞类', 'levir', REGIONS)
        result = select_batch([row('A', ['holes']), row('B'), row('C')], p)
        self.assertEqual([r['sample_id'] for r in result['rows']], ['A'])
        self.assertFalse(result['hardSatisfied'])

    def test_decreasing_count_after_replacement_never_returns_extra_rows(self):
        rows = [row(x) for x in 'ABCDE']
        old = compile_request('选4张复核', 'levir', REGIONS)
        middle = compile_request('第四张移出，再补一张', 'levir', REGIONS,
                                 previous={'plan': old, 'rows': rows[:4]})
        selected = select_batch(rows, middle)['rows']
        final = compile_request('改成2张，其他条件保持', 'levir', REGIONS,
                                previous={'plan': middle, 'rows': selected})
        result = select_batch(rows, final)
        self.assertEqual(len(result['rows']), 2)
        self.assertTrue(result['hardSatisfied'])

    def test_compound_request_keeps_quantity_region_exclusion_and_coverage(self):
        p = compile_request('LEVIR给我10张复核，排除已复核，兼顾碎片和空洞', 'whu', REGIONS)
        self.assertEqual((p['region_id'], p['count'], p['exclude_reviewed']), ('levir', 10, True))
        self.assertEqual(set(p['coverage']), {'fragmentation', 'holes'})
        self.assertFalse(p['issues'])

    def test_model_cannot_drop_or_change_explicit_constraints(self):
        p = compile_request('LEVIR给我10张复核，排除已复核，变化像素大于1000，空洞至少3张', 'whu', REGIONS,
            model_plan={'region_id': 'whu', 'count': 2, 'exclude_reviewed': False, 'coverage': [], 'minimums': {}, 'filters': []})
        self.assertEqual(p['region_id'], 'levir')
        self.assertEqual(p['count'], 10)
        self.assertTrue(p['exclude_reviewed'])
        self.assertEqual(p['minimums']['holes'], 3)
        self.assertEqual(p['filters'], [{'field': 'changed_px', 'op': 'gt', 'value': 1000}])
        self.assertTrue(p['preservedConstraints'])

    def test_model_can_understand_non_keyword_preferences(self):
        p = compile_request('帮我挑10张复核，像撒芝麻和蜂窝那样的都照顾到', 'levir', REGIONS,
            model_plan={'coverage': ['fragmentation', 'holes']})
        self.assertEqual(p['coverage'], ['fragmentation', 'holes'])

    def test_model_cannot_switch_the_selected_region_without_user_request(self):
        p = compile_request('帮我选10张复核', 'levir', REGIONS, model_plan={'region_id': 'whu'})
        self.assertEqual(p['region_id'], 'levir')

    def test_model_invented_hard_quota_does_not_execute(self):
        p = compile_request('选10张复核，兼顾空洞', 'levir', REGIONS,
                            model_plan={'minimums': {'holes': 7}})
        self.assertTrue(p['issues'])

    def test_only_an_explicit_request_releases_a_previous_family_minimum(self):
        old = compile_request('选10张复核，空洞至少3张', 'levir', REGIONS)
        p = compile_request('取消空洞最低数量，其他条件保持', 'levir', REGIONS,
                            previous={'plan': old, 'rows': []})
        self.assertNotIn('holes', p['minimums'])

    def test_unknown_tools_or_model_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            compile_request('给我10张复核', 'levir', REGIONS, model_plan={'sql': 'DELETE FROM samples'})

    def test_invented_model_threshold_requires_clarification(self):
        p = compile_request('帮我选10张复核', 'levir', REGIONS,
            model_plan={'filters': [{'field': 'changed_px', 'op': 'gt', 'value': 9000}]})
        self.assertTrue(p['issues'])

    def test_chinese_units_and_comparator_are_preserved(self):
        p = compile_request('选十张复核，变化像素至少一万，对象数量不超过3', 'levir', REGIONS)
        self.assertEqual(p['count'], 10)
        self.assertIn({'field': 'changed_px', 'op': 'ge', 'value': 10000}, p['filters'])
        self.assertIn({'field': 'object_count', 'op': 'le', 'value': 3}, p['filters'])

    def test_unsupported_time_area_or_semantic_filter_does_not_silently_disappear(self):
        for q in ['最近一周选10张复核', '给我10张超过100平方米的复核图', '只要道路的，选10张复核']:
            with self.subTest(q=q):
                self.assertTrue(compile_request(q, 'levir', REGIONS)['issues'])

    def test_numeric_or_is_not_replaced_with_and(self):
        p = compile_request('选10张复核，变化像素大于100或对象数量大于3', 'levir', REGIONS)
        self.assertTrue(p['issues'])

    def test_revision_keeps_all_other_constraints_and_the_unchanged_rows(self):
        old = compile_request('选3张复核，排除已复核，变化像素至少100，兼顾碎片和空洞', 'levir', REGIONS)
        previous = {'plan': old, 'rows': [row('A'), row('B'), row('C')]}
        p = compile_request('第三张移出，再补一张空洞类的', 'levir', REGIONS, previous=previous)
        self.assertEqual(p['excluded_ids'], ['C'])
        self.assertEqual(p['pinned_ids'], ['A', 'B'])
        self.assertEqual(p['replacement_family'], 'holes')
        self.assertEqual(p['filters'], old['filters'])
        self.assertEqual(p['count'], 3)

    def test_ordinal_in_a_ten_sample_revision_is_not_a_new_count(self):
        rows = [row(str(i)) for i in range(12)]
        old = compile_request('选10张复核', 'levir', REGIONS)
        plan = compile_request('第三张移出，再补一张', 'levir', REGIONS,
                               previous={'plan': old, 'rows': rows[:10]})
        self.assertEqual(plan['count'], 10)
        result = select_batch(rows, plan)
        self.assertEqual(len(result['rows']), 10)
        self.assertTrue(result['hardSatisfied'])

    def test_out_of_range_reference_stops_instead_of_using_selected_sample(self):
        previous = {'plan': compile_request('选3张复核', 'levir', REGIONS), 'rows': [row('A')]}
        self.assertTrue(compile_request('第三张换掉', 'levir', REGIONS, previous=previous)['issues'])

    def test_simple_rank_query_keeps_the_old_fast_path(self):
        self.assertFalse(is_review_request('当前区域变化像素前3名待复核样本'))
        self.assertTrue(is_review_request('给我10张复核，兼顾碎片和空洞'))
        self.assertTrue(is_review_request('第三张先移出，再补一张', has_previous=True))


class SelectionTests(unittest.TestCase):
    def plan(self, q='选3张复核，兼顾碎片和空洞'):
        return compile_request(q, 'levir', REGIONS)

    def test_coverage_is_prioritized_over_just_largest_area(self):
        rows = [row('large', px=9000), row('fragment', ['fragmentation'], px=300),
                row('hole', ['holes'], px=200), row('other', px=8000)]
        result = select_batch(rows, self.plan())
        self.assertEqual({r['sample_id'] for r in result['rows']}, {'fragment', 'hole', 'large'})
        self.assertTrue(result['hardSatisfied'])
        self.assertEqual(result['softUnmet'], [])

    def test_overlapping_coverage_is_counted_once_and_hard_minimums_are_feasible(self):
        rows = [row('a', ['fragmentation']), row('b', ['holes']), row('both', ['fragmentation', 'holes'])]
        p = self.plan('选1张复核，碎片至少1张，空洞至少1张')
        result = select_batch(rows, p)
        self.assertEqual([r['sample_id'] for r in result['rows']], ['both'])
        self.assertTrue(result['hardSatisfied'])

    def test_hard_quota_shortage_is_visible_and_never_filled_with_unflagged_rows(self):
        rows = [row('hole', ['holes']), row('normal'), row('another')]
        result = select_batch(rows, self.plan('选3张复核，空洞至少3张'))
        self.assertFalse(result['hardSatisfied'])
        self.assertEqual(result['availableCoverage']['holes'], 1)
        self.assertTrue(result['issues'])

    def test_confirmed_rows_are_excluded_and_low_risk_is_not_assumed_reviewed(self):
        rows = [row('done'), row('low', risk='low'), row('new')]
        result = select_batch(rows, self.plan('选2张复核，排除已复核'), reviewed_ids={'done'})
        self.assertEqual({r['sample_id'] for r in result['rows']}, {'low', 'new'})

    def test_missing_metrics_do_not_pass_a_zero_threshold(self):
        rows = [row('unknown', px=None), row('zero', px=0)]
        result = select_batch(rows, self.plan('选1张复核，变化像素等于0'))
        self.assertEqual([r['sample_id'] for r in result['rows']], ['zero'])

    def test_revision_replaces_only_one_sample_and_does_not_mark_it_reviewed(self):
        rows = [row('A'), row('B'), row('C'), row('D', ['holes'])]
        old = self.plan('选3张复核')
        p = compile_request('第三张移出，再补一张空洞类的', 'levir', REGIONS,
                            previous={'plan': old, 'rows': rows[:3]})
        self.assertEqual([r['sample_id'] for r in select_batch(rows, p)['rows']], ['A', 'B', 'D'])

    def test_soft_coverage_shortage_does_not_fabricate_an_error_diagnosis(self):
        result = select_batch([row('A'), row('B')], self.plan('选2张复核，兼顾空洞'))
        self.assertTrue(result['hardSatisfied'])
        self.assertEqual(result['softUnmet'], ['holes'])


class SnapshotTests(unittest.TestCase):
    def test_snapshot_extracts_public_fields_without_private_metrics_or_images(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'data.json'
            path.write_text(json.dumps({'regions': [{'id': 'levir', 'items': [{
                'id': 'A', 'changedPx': 0, 'objectCount': 0, 'reviewLevel': 'none',
                'maskF1': .99, 'assets': {'gtMask': 'secret.png'},
                'reviewReasonCodes': ['large_hole_ratio'],
                'maskQuality': {'hole_ratio': .4, 'border_contact': True}}]}]}), encoding='utf-8')
            source = Snapshot(path)
            data = source.rows('levir')[0]
            self.assertEqual(data['changed_px'], 0)
            self.assertIn('holes', data['families'])
            self.assertIn('border', data['families'])
            self.assertNotIn('maskF1', str(data))
            self.assertNotIn('secret.png', str(data))
            self.assertTrue(data['facts_hash'])


if __name__ == '__main__':
    unittest.main()
