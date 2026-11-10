"""Focused guards: identities, incomplete coverage, mapping, metrics and no-crop preprocessing."""
import unittest

import nbformat
import numpy as np
from PIL import Image

import model_a as a


class ModelATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = a.read_csv(a.LOCK)

    def test_real_frozen_inventory(self):
        six = a.verify_inventory(self.lock)
        self.assertEqual(len(six), 1800)
        self.assertEqual(six.groupby(['class_name', 'split']).size().value_counts().to_dict(), {240: 6, 60: 6})

    def test_duplicate_identity_and_content_rejected(self):
        for field in ['sample_id', 'video', 'video_sha256']:
            with self.subTest(field=field):
                bad = self.lock.copy()
                bad.loc[1, field] = bad.loc[0, field]
                with self.assertRaises(ValueError):
                    a.verify_inventory(bad)

    def test_label_and_external_split_mismatch_rejected(self):
        for field, value in [('class_id', '0'), ('class_name', 'saya'), ('source', 'test'), ('split', 'test')]:
            with self.subTest(field=field):
                bad = self.lock.copy()
                bad.loc[0, field] = value
                with self.assertRaises(ValueError):
                    a.verify_inventory(bad)

    def test_letterbox_retains_edges_rgb_and_determinism(self):
        pixels = np.zeros((100, 400, 3), dtype=np.uint8)
        pixels[:, :40] = (255, 0, 0)
        pixels[:, -40:] = (0, 0, 255)
        image = Image.fromarray(pixels)
        result = np.asarray(a.LetterboxRGB(224)(image))
        self.assertEqual(result.shape, (224, 224, 3))
        np.testing.assert_array_equal(result[0, 0], [114, 114, 114])
        np.testing.assert_array_equal(result[112, 1], [255, 0, 0])
        np.testing.assert_array_equal(result[112, 222], [0, 0, 255])
        x, y = a.transform()(image), a.transform()(image)
        self.assertTrue(a.torch.equal(x, y))
        self.assertEqual(tuple(x.shape), (3, 224, 224))
        self.assertEqual(x[0, 112, 1].item(), 1.)

    def test_alphabetical_mapping_and_invalid_mapping(self):
        self.assertEqual(a.label_mapping(dict(enumerate(sorted(a.CLASSES)))), [5, 3, 1, 4, 0, 2])
        with self.assertRaises(ValueError):
            a.label_mapping(dict(enumerate(a.ALL_CLASSES)))

    def test_metrics_known_answer_and_zero_support(self):
        metrics, per_class, matrix = a.classification_metrics([0, 0, 1, 2, 3, 4, 5], [0, 1, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(metrics['top1_accuracy'], 6/7)
        self.assertAlmostEqual(metrics['macro_f1'], 8/9)
        self.assertEqual(matrix[0, 1], 1)
        self.assertEqual(per_class.support.tolist(), [2, 1, 1, 1, 1, 1])
        metrics, _, _ = a.classification_metrics([0], [0])
        self.assertEqual(metrics['macro_f1'], 1/6)

    def test_unresolved_decisions_and_identity_edit_fail_closed(self):
        rows = a.read_csv(a.DECISIONS)
        inventory = a.read_csv(a.HERE / 'manifests/source_inventory.csv')
        catalogue = a.read_csv(a.CACHE / 'candidates.csv').set_index('candidate_id')
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            a.validate_decisions(rows, inventory, catalogue)
        with self.assertRaisesRegex(ValueError, 'every one'):
            a.validate_decisions(rows.iloc[:-1], inventory, catalogue)
        bad = rows.copy()
        bad.loc[0, 'split'] = 'val'
        with self.assertRaisesRegex(ValueError, 'frozen identity'):
            a.validate_decisions(bad, inventory, catalogue)

    def test_semantic_ambiguity_and_unbound_review_rejected(self):
        inventory = a.read_csv(a.HERE / 'manifests/source_inventory.csv')
        rows = a.read_csv(a.DECISIONS)
        rows['disposition'] = 'exclude'
        rows['decision_note'] = 'Test fixture only; never saved'
        rows['reviewer'] = 'fixture'
        index = rows[rows.candidate_id.ne('')].index[0]
        rows.loc[index, 'disposition'] = 'include'
        rows.loc[index, 'semantic_status'] = 'ambiguous'
        catalogue = a.read_csv(a.CACHE / 'candidates.csv').set_index('candidate_id')
        with self.assertRaisesRegex(ValueError, 'semantic review'):
            a.validate_decisions(rows, inventory, catalogue)
        rows.loc[index, 'semantic_status'] = 'valid_for_target'
        with self.assertRaisesRegex(ValueError, 'semantic review'):
            a.validate_decisions(rows, inventory, catalogue)
        rows.loc[index, 'reviewed_candidate_id'] = rows.loc[index, 'candidate_id']
        self.assertEqual(len(a.validate_decisions(rows, inventory, catalogue)), 1)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            a.safe_path(a.CACHE, '../../../../../outside.jpg')

    def test_notebook_schema_syntax_and_execution_flags(self):
        notebook = nbformat.read(a.HERE / 'Model_A_YOLO26_6Class_Training.ipynb', as_version=4)
        nbformat.validate(notebook)
        source = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
        self.assertIn('RUN_TRAINING = False', source)
        self.assertIn('RUN_SMOKE_TEST = False', source)
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'cell_{i}', 'exec')


if __name__ == '__main__':
    unittest.main(verbosity=2)
