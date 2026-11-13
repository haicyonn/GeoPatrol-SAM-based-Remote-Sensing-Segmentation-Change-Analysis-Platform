import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from inspection_agent import explain
from claim_validation import validate_claims


HITS = [{'title': '复核说明', 'section': '风险', 'text': '碎片化属于风险线索，需要人工检查。',
         'source': 'quality.md', 'version': 'v1'}]
ROWS = [{'sample_id': 'A', 'changed_px': 1000, 'object_count': 2},
        {'sample_id': 'B', 'changed_px': 900, 'object_count': 4}]


class ExplainClaimRegressionTests(unittest.TestCase):
    def test_numbers_from_another_sample_cannot_support_a_claim(self):
        result = explain('解释样本A', HITS, ROWS,
                         lambda _: '样本A变化900像素，有4个对象。[1]')
        self.assertEqual(result['generationStatus'], 'claim_fallback')
        self.assertNotIn('样本A变化900', result['answer'])

    def test_pixel_count_cannot_be_relabelled_as_ground_area(self):
        result = explain('解释样本A', HITS, ROWS,
                         lambda _: '样本A变化1000平方米。[1]')
        self.assertEqual(result['generationStatus'], 'claim_fallback')
        self.assertNotIn('1000平方米', result['answer'])

    def test_existing_citation_cannot_prove_universal_accuracy(self):
        result = explain('解释复核依据', HITS, ROWS,
                         lambda _: '依据资料，所有检测结果均准确，无需人工检查。[1]')
        self.assertEqual(result['generationStatus'], 'claim_fallback')
        self.assertNotIn('均准确', result['answer'])

    def test_metric_values_cannot_be_swapped_within_one_sample(self):
        result = explain('解释样本A', HITS, ROWS,
                         lambda _: '样本A变化2像素，有1000个对象。[1]')
        self.assertEqual(result['generationStatus'], 'claim_fallback')

    def test_valid_multi_sample_answer_preserves_each_binding(self):
        answer = '1. 样本A变化1,000像素，有2个对象。[1]\n2. 样本B变化900px，对象数量4个。[1]'
        result = explain('比较样本', HITS, ROWS, lambda _: answer)
        self.assertEqual(result['generationStatus'], 'generated')
        self.assertEqual(result['answer'], answer)
        self.assertFalse(result['validation']['scope']['semanticEntailment'])

    def test_unknown_numeric_values_keep_numeric_fallback(self):
        result = explain('解释样本', HITS, ROWS,
                         lambda _: '样本A变化999999像素。[1]')
        self.assertEqual(result['generationStatus'], 'numeric_fallback')

    def test_missing_or_invalid_citations_keep_citation_fallback(self):
        for answer in ['连通域需要检查。', '连通域需要检查。[99]']:
            with self.subTest(answer=answer):
                result = explain('解释', HITS, ROWS, lambda _: answer)
                self.assertEqual(result['generationStatus'], 'citation_fallback')

    def test_an_unbound_number_is_not_supported_by_a_matching_value(self):
        result = explain('解释复核', HITS, ROWS,
                         lambda _: '阈值为2。[1]')
        self.assertEqual(result['generationStatus'], 'claim_fallback')

    def test_ambiguous_current_sample_is_not_inferred_from_value(self):
        result = explain('解释', HITS, ROWS,
                         lambda _: '该样本变化900像素。[1]')
        self.assertEqual(result['generationStatus'], 'claim_fallback')

    def test_single_sample_can_bind_a_referential_subject(self):
        answer = '该样本变化1000像素，对象数量为2。[1]'
        result = explain('解释', HITS, ROWS[:1], lambda _: answer)
        self.assertEqual(result['generationStatus'], 'generated')

    def test_unknown_named_sample_cannot_borrow_the_only_row(self):
        result = explain('解释', HITS, ROWS[:1],
                         lambda _: '样本Z变化1000像素。[1]')
        self.assertEqual(result['generationStatus'], 'claim_fallback')

    def test_version_identifiers_and_model_names_are_not_numeric_metrics(self):
        hits = [{**HITS[0], 'version': '2026-10-25-v1.2'}]
        answer = '1. 来源版本2026-10-25-v1.2说明F1不代表单样本正确率。[1]'
        result = explain('解释', hits, [], lambda _: answer)
        self.assertEqual(result['generationStatus'], 'generated')

    def test_sample_identifiers_with_digits_are_not_numeric_metrics(self):
        rows = [{'sample_id': 'tile_2026_001', 'changed_px': 1000, 'object_count': 2}]
        answer = '样本tile_2026_001变化1000像素，有2个对象。[1]'
        result = explain('解释', HITS, rows, lambda _: answer)
        self.assertEqual(result['generationStatus'], 'generated')

    def test_report_counts_representatives_and_buckets_are_bound(self):
        facts = {'totalCount': 4, 'changedCount': 3, 'reviewCount': 2, 'correctionCount': 1,
                 'representatives': [{'stem': 'B', 'changedPx': 900, 'objectCount': 4}],
                 'riskBuckets': [{'key': 'high', 'label': '高风险复核', 'count': 2}]}
        answer = ('区域结论：共4张切片，其中变化切片3张，优先复核2张，高风险2张，'
                  '已有1条人工修订。代表样本B变化900像素，有4个对象。[1]')
        result = explain('报告', HITS, facts, lambda _: answer)
        self.assertEqual(result['generationStatus'], 'generated')

    def test_report_counts_cannot_borrow_another_metric(self):
        facts = {'totalCount': 4, 'changedCount': 3, 'reviewCount': 2}
        result = explain('报告', HITS, facts,
                         lambda _: '本区域共有2张切片，待复核4张。[1]')
        self.assertEqual(result['generationStatus'], 'claim_fallback')

    def test_bucket_counts_cannot_borrow_another_bucket(self):
        facts = {'totalCount': 4, 'riskBuckets': [{'key': 'high', 'count': 2},
                                                {'key': 'medium', 'count': 1}]}
        result = explain('报告', HITS, facts,
                         lambda _: '高风险1张，中风险2张。[1]')
        self.assertEqual(result['generationStatus'], 'claim_fallback')

    def test_report_percentage_uses_the_named_numerator_and_denominator(self):
        facts = {'totalCount': 4, 'changedCount': 3, 'reviewCount': 2}
        result = explain('报告', HITS, facts,
                         lambda _: '变化切片占75%，待复核样本占50%。[1]')
        self.assertEqual(result['generationStatus'], 'generated')

    def test_ratio_percent_conversion_is_explicit_and_metric_bound(self):
        facts = [{'sample_id': 'A', 'changed_ratio': 0.125, 'confidence': 0.9}]
        result = explain('解释', HITS, facts,
                         lambda _: '样本A变化比例12.5%，置信度90%。[1]')
        self.assertEqual(result['generationStatus'], 'generated')
        bad = explain('解释', HITS, facts,
                      lambda _: '样本A变化比例90%。[1]')
        self.assertEqual(bad['generationStatus'], 'claim_fallback')

    def test_exact_source_numeric_excerpt_requires_the_cited_source(self):
        hits = [HITS[0], {**HITS[0], 'text': '建议至少检查2次。'}]
        good = explain('解释', hits, [], lambda _: '建议至少检查2次。[2]')
        bad = explain('解释', hits, [], lambda _: '建议至少检查2次。[1]')
        self.assertEqual(good['generationStatus'], 'generated')
        self.assertEqual(bad['generationStatus'], 'claim_fallback')

    def test_a_negated_absolute_is_not_rejected_as_an_endorsement(self):
        answer = '不能认为所有检测结果均准确，也不代表无需人工检查。[1]'
        result = explain('解释', HITS, ROWS, lambda _: answer)
        self.assertEqual(result['generationStatus'], 'generated')

    def test_extractive_metadata_does_not_claim_validation_was_run(self):
        result = explain('解释', HITS, ROWS)
        self.assertIsNone(result['validation']['ok'])
        self.assertFalse(result['validation']['scope']['semanticEntailment'])

    def test_explicit_metric_cannot_be_overridden_by_a_conflicting_unit(self):
        for answer in ['样本A对象数量1000像素。[1]', '样本A的changed_px为2个对象。[1]']:
            with self.subTest(answer=answer):
                result = explain('解释', HITS, ROWS, lambda _: answer)
                self.assertEqual(result['generationStatus'], 'claim_fallback')

    def test_unknown_units_and_numeric_scale_suffixes_are_rejected(self):
        for answer in ['样本A变化1000km。[1]', '样本A变化1000亩。[1]',
                       '样本A变化1000万像素。[1]', '样本A变化1000kpx。[1]']:
            with self.subTest(answer=answer):
                result = explain('解释', HITS, ROWS, lambda _: answer)
                self.assertEqual(result['generationStatus'], 'claim_fallback')

    def test_a_multi_sample_total_cannot_borrow_the_last_sample_value(self):
        result = explain('解释', HITS, ROWS,
                         lambda _: '样本A和样本B合计变化900像素。[1]')
        self.assertEqual(result['generationStatus'], 'claim_fallback')

    def test_raw_ratios_are_not_silently_rounded(self):
        facts = [{'sample_id': 'A', 'changed_ratio': 0.125}]
        result = explain('解释', HITS, facts,
                         lambda _: '样本A变化比例0.1。[1]')
        self.assertNotEqual(result['generationStatus'], 'generated')

    def test_many_decimal_places_do_not_crash_validation(self):
        answer = '样本A变化1000.' + '0' * 40 + '像素。[1]'
        result = explain('解释', HITS, ROWS, lambda _: answer)
        self.assertEqual(result['generationStatus'], 'generated')

    def test_chinese_numeric_measurements_keep_the_same_sample_binding(self):
        good = explain('解释', HITS, ROWS,
                       lambda _: '样本A变化一千像素，有两个对象。[1]')
        bad = explain('解释', HITS, ROWS,
                      lambda _: '样本A变化九百像素，有四个对象。[1]')
        self.assertEqual(good['generationStatus'], 'generated')
        self.assertEqual(bad['generationStatus'], 'claim_fallback')

    def test_unitless_chinese_metric_values_do_not_bypass_checks(self):
        good = explain('解释', HITS, ROWS, lambda _: '样本A对象数量为二。[1]')
        bad = explain('解释', HITS, ROWS, lambda _: '样本A对象数量为四。[1]')
        self.assertEqual(good['generationStatus'], 'generated')
        self.assertEqual(bad['generationStatus'], 'claim_fallback')

    def test_chinese_percentage_is_a_ratio_not_an_unchecked_phrase(self):
        facts = [{'sample_id': 'A', 'changed_ratio': 0.125, 'confidence': 0.9}]
        good = explain('解释', HITS, facts,
                       lambda _: '样本A变化比例百分之十二点五。[1]')
        bad = explain('解释', HITS, facts,
                      lambda _: '样本A变化比例百分之九十。[1]')
        self.assertEqual(good['generationStatus'], 'generated')
        self.assertEqual(bad['generationStatus'], 'claim_fallback')

    def test_public_validator_reports_the_failed_binding_and_its_scope(self):
        result = validate_claims('样本A变化900像素。[1]', HITS, ROWS)
        self.assertFalse(result['ok'])
        self.assertEqual(result['reasons'][0]['code'], 'sample_metric_mismatch')
        self.assertEqual(result['reasons'][0]['sample'], 'A')
        self.assertEqual(result['reasons'][0]['metric'], 'changed_px')
        self.assertEqual(result['reasons'][0]['unit'], 'pixels')
        self.assertEqual(result['scope']['checkedNumericClaims'], 1)
        self.assertFalse(result['scope']['fullSemanticCorrectness'])

    def test_units_before_the_value_are_checked_too(self):
        for answer in ['样本A变化面积平方米为1000。[1]', '样本A变化百分比为1000。[1]']:
            with self.subTest(answer=answer):
                result = explain('解释', HITS, ROWS, lambda _: answer)
                self.assertEqual(result['generationStatus'], 'claim_fallback')

    def test_source_citation_cannot_leak_from_the_previous_sentence(self):
        hits = [{**HITS[0], 'text': '建议检查3次。'},
                {**HITS[0], 'text': '建议检查2次。'}]
        result = explain('解释', hits, [],
                         lambda _: '建议检查2次。[2]建议检查2次。[1]')
        self.assertEqual(result['generationStatus'], 'claim_fallback')

    def test_numeric_sample_id_is_not_a_blanket_exemption_for_its_value(self):
        rows = [{'sample_id': '123', 'changed_px': 1000, 'object_count': 2}]
        good = explain('解释', HITS, rows,
                       lambda _: '样本123变化1000像素，有2个对象。[1]')
        bad = explain('解释', HITS, rows,
                      lambda _: '样本123变化123像素。[1]')
        self.assertEqual(good['generationStatus'], 'generated')
        self.assertEqual(bad['generationStatus'], 'numeric_fallback')

    def test_numeric_sample_id_cannot_hide_another_samples_wrong_object_count(self):
        rows = [{'sample_id': 'A', 'changed_px': 1000, 'object_count': 4},
                {'sample_id': '2', 'changed_px': 900, 'object_count': 2}]
        for answer in ['样本A变化1000像素，有2个对象。[1]', '样本A变化1000像素，2个对象。[1]']:
            with self.subTest(answer=answer):
                result = explain('解释', HITS, rows, lambda _: answer)
                self.assertEqual(result['generationStatus'], 'claim_fallback')


if __name__ == '__main__':
    unittest.main()
