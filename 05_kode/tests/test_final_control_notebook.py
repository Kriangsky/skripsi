"""TASK 0 notebook contract: safe defaults and no scientific execution paths."""
import ast
from pathlib import Path
import unittest

import nbformat


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT/'06_notebook/eksperimen/BISINDO_FINAL_EXPERIMENT_CONTROL.ipynb'


class FinalControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(NOTEBOOK, 4)

    def test_schema_sections_and_python(self):
        nbformat.validate(self.notebook)
        sections = [c for c in self.notebook.cells
                    if c.cell_type == 'markdown' and c.source.startswith('## ')]
        self.assertEqual(len(sections), 12)
        for cell in sections:
            for field in ['Status:', 'Tujuan:', 'Input/dependensi:', 'Output:',
                          'Kriteria penerimaan:', 'Risiko:', 'Perubahan bukti ilmiah:']:
                self.assertIn(field, cell.source)
        for cell in self.notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, 'final-control-cell', 'exec')

    def test_default_flags_and_opt_in_refused(self):
        setup = next(c.source for c in self.notebook.cells if 'setup' in c.metadata.get('tags', []))
        assignments = {node.targets[0].id: node.value.value for node in ast.parse(setup).body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                       and isinstance(node.value, ast.Constant)}
        for flag, tag in [('RUN_TRAINING', 'training-gate'), ('RUN_FREEZE', 'freeze-gate'),
                          ('RUN_EXTERNAL_SCORING', 'external-gate')]:
            self.assertIs(assignments[flag], False)
            source = next(c.source for c in self.notebook.cells if tag in c.metadata.get('tags', []))
            exec(source, {flag: False})
            with self.assertRaises(PermissionError):
                exec(source, {flag: True})

    def test_no_training_inference_freeze_or_algorithm_copy(self):
        forbidden = {'train', 'prepare', 'probe', 'evaluate', 'score', 'freeze', 'predict',
                     'development', 'bootstrap', 'export_reports', 'YOLO', 'load',
                     'system', 'run', 'Popen', 'write_text', 'write_bytes', 'to_csv', 'save'}
        for cell in self.notebook.cells:
            if cell.cell_type != 'code':
                continue
            tree = ast.parse(cell.source)
            self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.ClassDef))
                                 for node in ast.walk(tree)))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
                    self.assertNotIn(name, forbidden)


if __name__ == '__main__':
    unittest.main()
