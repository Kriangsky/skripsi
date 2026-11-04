import ast
import hashlib
import json
from pathlib import Path
import unittest

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / 'BISINDO_YOLO26_Vocabulary_Expansion_Research.ipynb'


class StageZeroResearchPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(NOTEBOOK, 4)
        cls.plan = cls.notebook.metadata['bisindo_stage0']

    def test_notebook_schema_kernel_and_python_syntax(self):
        nbformat.validate(self.notebook)
        self.assertEqual(self.notebook.metadata.kernelspec.name, 'thesis_cuthbert')
        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'stage0-cell-{i}', 'exec')

    def test_candidate_hash_matches_document_and_notebook(self):
        digest = hashlib.sha256((ROOT / 'PREREGISTRATION.md').read_bytes()).hexdigest()
        self.assertEqual(digest, self.plan['preregistration_sha256'])
        self.assertEqual(digest, (ROOT / 'PREREGISTRATION.sha256').read_text().split()[0])

    def test_training_disabled_and_notebook_has_no_execution_api(self):
        self.assertEqual(self.plan['stage'], 0)
        self.assertFalse(self.plan['training_enabled'])
        self.assertEqual(self.plan['status'], 'AWAITING_STAGE0_APPROVAL')
        prohibited = {'train', 'fit', 'run', 'Popen', 'system', 'exec', 'eval',
                      'write_text', 'write_bytes', 'to_csv', 'unlink', 'mkdir', 'open'}
        for cell in self.notebook.cells:
            if cell.cell_type != 'code':
                continue
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, 'id', getattr(node.func, 'attr', ''))
                    self.assertNotIn(name, prohibited)

    def test_registered_hypotheses_seeds_and_grid(self):
        self.assertEqual(self.plan['delta'], .02)
        self.assertEqual(self.plan['h2_accuracy_per_class'], .80)
        self.assertEqual(self.plan['seeds'], [42, 43, 44, 45, 46])
        self.assertEqual(self.plan['sampling_seed'], 42)
        self.assertEqual(self.plan['checkpoint'], 'last.pt')
        grid = self.plan['grid']
        self.assertEqual(len(grid), 9)
        self.assertEqual(len({r['id'] for r in grid}), 9)
        self.assertEqual({r['imgsz'] for r in grid}, {224, 320, 448})
        self.assertEqual({r['epochs'] for r in grid}, {10, 15, 30})

    def test_existing_artifacts_unchanged(self):
        protected = self.plan['protected_artifact_sha256']
        self.assertIn('experiment_yolo26_v3/split_lock.csv', protected)
        self.assertIn('classification_pilot.py', protected)
        self.assertIn('BISINDO_YOLO26_Classification_End_to_End.ipynb', protected)
        for name, digest in protected.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_existing_run_records_support_snapshot(self):
        work = ROOT / 'experiment_yolo26_classification_pilot'
        record = json.loads((work / 'runs/trained_models.json').read_text())
        self.assertEqual({r['epoch'] for r in record['models']}, {30})
        self.assertEqual({r['model']: r['actual_optimizer_steps'] for r in record['models']},
                         {'A': 690, 'B': 900})
        self.assertEqual(len({r['initial_checkpoint_sha256'] for r in record['models']}), 1)


if __name__ == '__main__':
    unittest.main()
