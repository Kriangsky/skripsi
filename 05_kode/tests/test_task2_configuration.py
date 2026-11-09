"""TASK2 artifact contract: proposals cannot authorize execution or rewrite evidence."""
import ast
import json
import math
from pathlib import Path
import unittest

import nbformat
import classification_research as research

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'01_hasil/addendum/scientific_decisions_task2_20261109'


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf8'))


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate=read('candidate_config.json')
        cls.audit=read('effective_configuration.json')

    def test_candidate_not_approval_or_frozen_protocol(self):
        c=self.candidate
        self.assertEqual(c['schema_version'],1)
        self.assertEqual(c['kind'],'CANDIDATE')
        self.assertEqual(c['status'],'PROPOSED_NOT_APPROVED')
        for key in ['training_authorized','freeze_authorized','external_scoring_authorized','protocol_amended']:
            self.assertIs(c[key],False)
        self.assertIsNone(c['approval'])
        self.assertEqual(c['readiness']['status'],'BLOCKED')
        self.assertIsNone(c['data']['internal_manifest'])
        self.assertIsNone(c['data']['external_manifest'])

    def test_same_baseline_training_across_arms_no_search(self):
        c=self.candidate
        baseline=dict(research.base.CONFIG);baseline.pop('seed')
        self.assertEqual(c['training'],baseline)
        self.assertEqual(c['training_seeds'],[42,43,44,45,46])
        self.assertEqual(c['class_mapping'],list(research.base.CLASSES))
        self.assertEqual(c['initial_checkpoint']['sha256'],research.PINNED_SHA)
        self.assertTrue(c['independent_initialization'])
        self.assertEqual(set(c['arms']),{'A','B'})
        self.assertEqual(c['checkpoint_selection']['file'],'last.pt')
        for key in ['best_checkpoint_selection','early_stopping','best_seed_selection','resume']:
            self.assertIs(c['checkpoint_selection'][key],False)

    def test_full_data_counts_and_steps_are_planned_not_measured(self):
        c=self.candidate
        self.assertEqual(c['data']['targets'],dict(train=240,val=60))
        for arm,n in [('A',6),('B',8)]:
            data=c['arms'][arm]
            self.assertEqual(data['class_ids'],list(range(n)))
            self.assertEqual(data['planned_train'],240*n)
            self.assertEqual(data['planned_val'],60*n)
            batches=math.ceil(data['planned_train']/c['training']['batch'])
            self.assertEqual(data['planned_batches_per_epoch'],batches)
            self.assertEqual(data['planned_optimizer_steps'],batches*30)
            self.assertEqual(data['planned_warmup_iterations'],batches*3)
        self.assertNotEqual(c['arms']['A']['planned_optimizer_steps'],c['arms']['B']['planned_optimizer_steps'])
        self.assertTrue(c['budget']['actual_counts_require_runtime_verification'])
        self.assertEqual(c['data']['split_lock_sha256'],research.base.b.sha(ROOT/c['data']['split_lock']))

    def test_no_stochastic_transforms_or_normalization_drift(self):
        p=self.candidate['preprocessing']
        self.assertEqual(p['imgsz'],224)
        self.assertEqual(p['padding'],114)
        self.assertEqual(p['color'],'RGB')
        self.assertEqual(p['normalize_mean'],[0,0,0])
        self.assertEqual(p['normalize_std'],[1,1,1])
        self.assertTrue(p['preserve_aspect_ratio'])
        self.assertTrue(p['train_val_external_identical'])
        for key in ['crop','horizontal_flip','vertical_flip','randaugment','random_erasing']:
            self.assertIs(p[key],False)

    def test_ten_actual_runs_have_transform_step_and_lr_evidence(self):
        runs=self.audit['runs']
        self.assertEqual({(r['seed'],r['arm']) for r in runs},{(s,a) for s in range(42,47) for a in ['A','B']})
        self.assertEqual(len(runs),10)
        for r in runs:
            self.assertEqual(r['checkpoint_version'],'8.4.197')
            self.assertEqual(r['steps'],690 if r['arm']=='A' else 900)
            self.assertEqual(r['batches_per_epoch'],23 if r['arm']=='A' else 30)
            self.assertTrue(r['synthetic_transform_exact_rgb_div255'])
            self.assertTrue(r['transform_repeats_identical'])
            self.assertTrue(r['all_epoch_lr_matches_local_source_reconstruction'])
            self.assertEqual(r['checkpoint_result_epochs'],list(range(1,31)))
            self.assertEqual(r['checkpoint_epoch'],-1)
            for lr in r['epoch30_learning_rates']:
                self.assertAlmostEqual(lr,.00043)
            for forbidden in ['Crop','Flip','RandAugment','RandomErasing']:
                self.assertNotIn(forbidden,r['checkpoint_transform'])
        self.assertFalse(self.audit['historical_framework_source_hash_available'])

    def test_h2_verbatim_and_no_fabricated_approvals(self):
        d=read('scientific_decisions.json')
        text=(ROOT/'07_dokumen/protokol/PREREGISTRATION.md').read_text(encoding='utf8')
        expected=text[text.index('- H2 kemampuan baru:'):text.index('\n- H3 sekunder:')]
        self.assertEqual(d['h2_verbatim'],expected)
        self.assertEqual(d['recommended_h1'],'B')
        self.assertIsNone(d['approved_h1'])
        self.assertIsNone(d['academic_approval_evidence'])
        self.assertIsNone(d['signatures'])
        self.assertFalse(d['protocol_amended'])
        self.assertFalse(d['task3_authorized'])
        self.assertEqual(self.candidate['decisions']['h2_threshold'],.80)

    def test_seed_provenance_preserves_uncertainty(self):
        s=read('seed_provenance.json')
        self.assertEqual(s['seeds'],[42,43,44,45,46])
        self.assertEqual(s['excluded_registered_seeds'],[])
        self.assertIsNone(s['rationale_for_42_or_consecutive_numbers'])
        self.assertIsNone(s['exact_selection_time'])
        self.assertTrue(s['formal_candidate']['pilot42_already_observed'])
        self.assertFalse(s['complete_omitted_attempt_attestation'])

    def test_notebook_historical_code_cells_preserved_and_new_cells_read_only(self):
        before=nbformat.from_dict(read('notebook_before.json'))
        current=nbformat.read(ROOT/'06_notebook/eksperimen/BISINDO_FINAL_EXPERIMENT_CONTROL.ipynb',4)
        by_id={c.id:c for c in current.cells}
        for cell in before.cells:
            if cell.cell_type=='code':
                for key in ['source','outputs','execution_count']:
                    self.assertEqual(by_id[cell.id][key],cell[key])
        fresh=[c for c in current.cells if any(t.startswith('task2-') for t in c.metadata.get('tags',[]))]
        self.assertEqual(len(fresh),2)
        allowed={'read','DataFrame','display','print','items'}
        for cell in fresh:
            tree=ast.parse(cell.source)
            self.assertFalse(any(isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom)) for n in ast.walk(tree)))
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    self.assertIn(getattr(node.func,'attr',getattr(node.func,'id','')),allowed)


if __name__=='__main__':
    unittest.main()
