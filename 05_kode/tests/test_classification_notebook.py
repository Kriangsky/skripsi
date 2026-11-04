import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / 'BISINDO_YOLO26_Classification_End_to_End.ipynb'


class ClassificationNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(NOTEBOOK, 4)

    def helper(self, work):
        cell = next(c for c in self.notebook.cells if 'cli-helper' in c.metadata.get('tags', []))
        namespace = dict(ROOT=ROOT, SOURCE=ROOT / 'experiment_yolo26_v3', WORK=work, Path=Path, subprocess=subprocess,
                         sys=sys, json=json, CLI_HISTORY=[])
        exec(compile(cell.source, 'notebook-helper', 'exec'), namespace)
        return namespace

    def test_schema_kernel_and_syntax(self):
        nbformat.validate(self.notebook)
        self.assertEqual(self.notebook.metadata.kernelspec.name, 'thesis_cuthbert')
        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'cell-{i}', 'exec')

    def test_partial_runs_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            helper = self.helper(work)
            self.assertTrue(helper['should_execute'](work, True))
            (work / 'runs/model_A').mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                helper['should_execute'](work, True)
            self.assertFalse(helper['should_execute'](work, False))

    def test_completed_runs_are_not_retrained(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work / 'evaluation').mkdir()
            (work / 'runs').mkdir()
            (work / 'evaluation/evaluation_complete.json').write_text('{}')
            (work / 'runs/trained_models.json').write_text('{}')
            self.assertFalse(self.helper(work)['should_execute'](work, True))

    def test_cli_errors_propagate_with_real_return_code(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            helper = self.helper(work)
            with patch.object(subprocess, 'Popen') as factory:
                process = factory.return_value
                process.stdout = iter(['sample log\n'])
                process.wait.return_value = 2
                process.poll.return_value = 2
                with self.assertRaises(subprocess.CalledProcessError):
                    helper['run_cli']('probe')
            self.assertEqual(helper['CLI_HISTORY'][0]['returncode'], 2)
            self.assertEqual(factory.call_args.args[0][0], sys.executable)
            self.assertFalse(factory.call_args.kwargs.get('shell', False))
            self.assertTrue((work / 'reports/notebook_probe.log').is_file())

    def test_helper_stops_its_child_on_interrupt(self):
        with tempfile.TemporaryDirectory() as temp:
            helper = self.helper(Path(temp))
            with patch.object(subprocess, 'Popen') as factory:
                process = factory.return_value
                process.stdout.__iter__.side_effect = KeyboardInterrupt
                process.poll.return_value = None
                with self.assertRaises(KeyboardInterrupt):
                    helper['run_cli']('train')
                process.terminate.assert_called_once()

    def test_no_raw_subject_images_embedded(self):
        sources = '\n'.join(c.source for c in self.notebook.cells)
        self.assertNotIn("Image(filename=str(WORK/'selected_frames'", sources)
        self.assertIn('predictions.csv', sources)
        self.assertIn('notebook_execution_manifest.json', sources)


if __name__ == '__main__':
    unittest.main()
