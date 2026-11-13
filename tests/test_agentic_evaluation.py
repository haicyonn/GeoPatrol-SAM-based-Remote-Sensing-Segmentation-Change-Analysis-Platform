import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
try:
    import evaluate_agentic as evaluation
except ImportError:
    evaluation = None


class EvaluationMetricTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(evaluation, 'Comparison evaluator missing')

    def test_missing_labels_are_zero_and_rank_two_is_half_reciprocal(self):
        rows = [{'expectedSection': 'A', 'sections': ['B', 'A'], 'elapsedMs': 10},
                {'expectedSection': 'C', 'sections': ['D'], 'elapsedMs': 30}]
        metrics = evaluation.retrieval_metrics(rows)
        self.assertEqual(metrics['hitAt1'], 0)
        self.assertEqual(metrics['hitAt3'], 0.5)
        self.assertEqual(metrics['mrrAt6'], 0.25)

    def test_empty_evaluation_does_not_claim_perfect_quality(self):
        metrics = evaluation.retrieval_metrics([])
        self.assertEqual(metrics['total'], 0)
        self.assertIsNone(metrics['hitAt3'])


if __name__ == '__main__':
    unittest.main()
