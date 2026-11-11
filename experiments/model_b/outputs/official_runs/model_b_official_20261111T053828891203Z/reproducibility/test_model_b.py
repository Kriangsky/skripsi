import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_b_support as b


class ModelBTests(unittest.TestCase):
    def test_full_predictions_count_new_intrusions_as_false_negatives(self):
        metrics, per_class, matrix = b.metrics([0, 1, 2, 3, 4, 5], [6, 1, 2, 3, 4, 5], labels=range(6))
        self.assertAlmostEqual(metrics['macro_f1'], 5 / 6)
        self.assertEqual(matrix[0, 6], 1)
        self.assertEqual(per_class.iloc[0].recall, 0)

    def test_mapping_and_probability_guard(self):
        self.assertEqual(b.mapping(dict(enumerate(sorted(b.CLASSES)))), [5, 3, 6, 1, 4, 0, 2, 7])
        with self.assertRaises(ValueError):
            b.mapping(dict(enumerate(b.CLASSES[:6])))
        with self.assertRaises(ValueError):
            b.metrics([0], [8])

    def test_dataset_shared_identity_and_frozen_splits(self):
        rows, audit = b.audit_dataset(verify_files=False)
        self.assertEqual(len(rows), 2400)
        self.assertEqual(audit['shared_samples_exact'], 1800)
        self.assertEqual(rows.groupby(['class_name', 'split']).size().value_counts().to_dict(), {240: 8, 60: 8})

    def test_validator_fitness_is_macro_f1(self):
        import torch
        validator = b.MacroF1Validator.__new__(b.MacroF1Validator)
        from ultralytics.utils.metrics import ClassifyMetrics
        validator.metrics = ClassifyMetrics()
        validator.targets = [torch.arange(8)]
        validator.pred = [torch.tensor([[6, 0, 1, 2, 3]] + [[i, 0, 1, 2, 3] for i in range(1, 8)])]
        stats = validator.get_stats()
        self.assertEqual(stats['fitness'], stats['metrics/macro_f1'])
        self.assertNotEqual(stats['fitness'], stats['metrics/accuracy_top1'])


if __name__ == '__main__':
    unittest.main()
