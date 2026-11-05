import copy
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import confirmatory_statistics as s
import confirmatory_evaluation as e
import interference_metrics as m
from test_interference_metrics import predictions


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_reproducibility_and_f1_recomputed(self):
        y = np.repeat(np.arange(6), 3); a = y[None, :]; b = a.copy(); b[0, 0] = 6
        r = s.bootstrap(y, a, b, repetitions=200, seed=7)
        rr = s.bootstrap(y, a, b, repetitions=200, seed=7)
        np.testing.assert_array_equal(r['hierarchical'], rr['hierarchical'])
        rng = np.random.default_rng(7)
        ids = np.concatenate([rng.choice(np.flatnonzero(y == k), 3, replace=True) for k in range(6)])
        expected = m.metrics(y[ids], b[0, ids], range(6))['macro_f1'] - 1
        self.assertAlmostEqual(r['video_only'][0], expected)
        self.assertEqual(r['n_videos'], 18)
        self.assertTrue((r['hierarchical'] <= 0).all())

    def test_multiple_seed_mean_not_pooled_trials(self):
        y = np.repeat(np.arange(6), 2); a = np.tile(y, (3, 1)); b = a.copy(); b[1, :2] = 6
        r = s.bootstrap(y, a, b, repetitions=100)
        expected = np.mean([m.metrics(y, x, range(6))['macro_f1'] - 1 for x in b])
        self.assertAlmostEqual(r['estimate'], expected)
        self.assertEqual(r['n_videos'], 12)
        self.assertEqual(r['n_seeds'], 3)
        self.assertGreater(r['seed_delta_sd'], 0)
        z = s.bootstrap(y, a, a, repetitions=100)
        np.testing.assert_array_equal(z['hierarchical'], np.zeros(100))
        with self.assertRaises(ValueError): s.bootstrap(y[:-1], a, b, repetitions=100)

    def test_ni_boundary_and_unresolved_margin(self):
        self.assertEqual(s.non_inferiority(-.02, .02, 'substantive source'), 'non-inferiority not demonstrated')
        self.assertEqual(s.non_inferiority(-.019, .02, 'substantive source'), 'non-inferiority demonstrated')
        self.assertEqual(s.non_inferiority(-.01, None, None), 'NI_MARGIN_UNRESOLVED')
        with self.assertRaises(ValueError): s.non_inferiority(0, .02, '')
        with self.assertRaises(ValueError): s.non_inferiority(float('nan'), .02, 'source')

    def test_design_effect_and_simultaneous_intervals(self):
        r = s.design_effect_sensitivity(0, np.array([-.1, 0, .1]))
        self.assertEqual([x['design_effect'] for x in r], [1, 1.5, 2])
        self.assertGreater(r[2]['ci95_high'], r[0]['ci95_high'])
        draws = np.tile(np.linspace(-1, 1, 100)[:, None], (1, 6))
        intervals = s.simultaneous_recall_intervals(draws)
        self.assertEqual(len(intervals), 6)
        self.assertLessEqual(intervals[0][0], np.quantile(draws[:, 0], .025))

    def test_premature_external_scoring_rejected_before_inference(self):
        plan = dict(status='draft', ni_margin=None)
        with self.assertRaises(ValueError): e.validate_plan(plan, Path('.'))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with self.assertRaises((ValueError, FileNotFoundError)): e.score(p/'missing.json', p/'approval.json', p/'out')
            self.assertFalse((p/'out').exists())

    def test_development_exports_and_all_seeds_enforced(self):
        plan = dict(training_seeds=[42,43],bootstrap_replicates=100,bootstrap_seed=7,
                    h2_threshold=.8,ni_status='NI_MARGIN_UNRESOLVED',ni_margin=None,ni_justification=None)
        frames = {seed:dict(A=predictions(6,'A',seed),B=predictions(8,'B',seed)) for seed in [42,43]}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out'
            e.export_reports(frames,plan,out,'development')
            import pandas as pd
            self.assertEqual(len(list(out.glob('*.csv'))),8)
            primary=pd.read_csv(out/'common6_primary_results.csv')
            self.assertEqual(primary.iloc[-1].ni_decision,'not tested: development evidence')
            self.assertEqual(primary.iloc[-1].delta,0)
            detail=json.loads((out/'analysis_details.json').read_text())
            self.assertEqual(detail['n_videos'],6)
            with self.assertRaises(FileExistsError): e.export_reports(frames,plan,out,'development')
            with self.assertRaises(ValueError): e.export_reports({42:frames[42]},plan,Path(tmp)/'bad','development')
            bad=copy.deepcopy(frames); bad[43]['B'].loc[0,'image_sha256']='changed'
            with self.assertRaises(ValueError): e.export_reports(bad,plan,Path(tmp)/'bad','development')

    def test_freeze_approval_binding_drift_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); freeze_path=root/'freeze.json'; asset=root/'asset.txt'; asset.write_text('original')
            plan=root/'plan.json'; plan.write_text('{}')
            record=dict(status='frozen',files_sha256={str(asset):e.sha(asset)},plan_path=str(plan),plan_sha256=e.sha(plan))
            e.write_new(freeze_path,record)
            with self.assertRaises(FileExistsError): e.write_new(freeze_path,record)
            approval=root/'approval.json'
            data=dict(action='final_external_scoring',approved=True,reviewer='synthetic reviewer',
                      timestamp='synthetic',freeze_sha256=e.sha(freeze_path))
            e.write_new(approval,data)
            e.verify_freeze(freeze_path,approval)
            asset.write_text('changed')
            with self.assertRaises(ValueError): e.verify_freeze(freeze_path,approval)
            data['freeze_sha256']='wrong'; approval.write_text(json.dumps(data))
            with self.assertRaises(PermissionError): e.verify_freeze(freeze_path,approval)

    def test_draft_actual_plan_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'freeze.json'
            with self.assertRaises(ValueError): e.freeze(e.DEFAULT_PLAN,out)
            self.assertFalse(out.exists())

    def test_final_gate_checks_provenance_exposure_validation_and_seed_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan=copy.deepcopy(e.read(e.DEFAULT_PLAN))
            plan.update(status='final',ni_status='NI_MARGIN_JUSTIFIED',ni_margin=.02,
                        ni_justification='synthetic substantive rationale',plan_path='plan.json')
            for key in plan['artifacts']:
                plan['artifacts'][key]=key+'.json'
                (root/(key+'.json')).write_text('{}')
            config=dict(status='final',task='classify',checkpoint_rule='last.pt',imgsz=2,epochs=30,
                        preprocessing='frozen_letterbox_RGB_255_gray114_no_crop_no_flip')
            (root/'experiment_config.json').write_text(json.dumps(config))
            for seed in plan['training_seeds']:
                for name in ['A','B']:
                    ck=root/f'{seed}_{name}'/'last.pt'; ck.parent.mkdir(); ck.write_bytes(b'synthetic checkpoint')
                    plan['checkpoints'].append(dict(seed=seed,model=name,path=str(ck),sha256=e.sha(ck),tier='final',
                        initial_checkpoint_sha256=e.INITIAL_SHA,independent_initialization=True,
                        experiment_config_sha256=e.sha(root/'experiment_config.json')))
            (root/'plan.json').write_text(json.dumps(plan))
            evidence={
                'scientific_decision':dict(status='approved',reviewer='synthetic',timestamp='synthetic',
                    plan_sha256=e.sha(root/'plan.json'),addendum_sha256=e.sha(root/'addendum.json'),unresolved_conflicts=[]),
                'provenance_review':dict(status='verified',different_from_internal_signers=True,evidence=['synthetic'],reviewer='synthetic'),
                'exposure_review':dict(status='reviewed',used_for_tuning=False,prior_scoring=False,history=['preprocessed only'],reviewer='synthetic'),
                'validation_record':dict(status='passed',external_scoring_performed=False,source_sha256={},tests=[dict(exit_code=0)]*2)}
            for name,data in evidence.items(): (root/(name+'.json')).write_text(json.dumps(data))
            with patch.object(e,'source_hashes',return_value={}), patch.object(e,'INITIAL_SHA',e.sha(root/'initial_checkpoint.json')):
                # Match synthetic initialization metadata, without reading/deserializing any real checkpoint.
                for item in plan['checkpoints']: item['initial_checkpoint_sha256']=e.INITIAL_SHA
                paths,_=e.validate_plan(plan,root)
                self.assertEqual(len([k for k in paths if k.startswith('checkpoint_')]),10)
                for name,key,value in [('provenance_review','different_from_internal_signers',False),
                                       ('exposure_review','used_for_tuning',True),('validation_record','status','failed')]:
                    bad=copy.deepcopy(evidence[name]); bad[key]=value
                    (root/(name+'.json')).write_text(json.dumps(bad))
                    with self.assertRaises(ValueError): e.validate_plan(plan,root)
                    (root/(name+'.json')).write_text(json.dumps(evidence[name]))
                bad=copy.deepcopy(plan); bad['checkpoints'].pop()
                with self.assertRaises(ValueError): e.validate_plan(bad,root)

    def test_actual_file_duplicate_and_split_checks_on_synthetic_manifests(self):
        import pandas as pd
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=[]
            for i in range(408):
                internal=i<8; cid=i if internal else (i-8)//50
                video=root/f'{i}.mp4'; video.write_bytes(f'synthetic video {i}'.encode())
                image=root/f'{i}.png'; Image.new('RGB',(2,2),(i%256,i//256,0)).save(image)
                rows.append(dict(sample_id=str(i),video=str(video),video_sha256=e.sha(video),image_path=str(image),
                                 image_sha256=e.sha(image),class_id=cid,class_name=m.ALL_CLASSES[cid],
                                 source='train' if internal else 'test',split='train' if internal else 'test'))
            paths={name:root/f'{name}.csv' for name in ['internal_manifest','external_manifest','split_mapping']}
            pd.DataFrame(rows[:8]).to_csv(paths['internal_manifest'],index=False)
            pd.DataFrame(rows[8:]).to_csv(paths['external_manifest'],index=False)
            pd.DataFrame(rows).to_csv(paths['split_mapping'],index=False)
            files=e.checked_manifests(paths,root,dict(imgsz=2))
            self.assertEqual(len(files),816)
            # Identical decoded pixels even with distinct video bytes must fail.
            Image.new('RGB',(2,2),(0,0,0)).save(root/'8.png')
            rows[8]['image_sha256']=e.sha(root/'8.png')
            pd.DataFrame(rows[8:]).to_csv(paths['external_manifest'],index=False)
            with self.assertRaises(ValueError): e.checked_manifests(paths,root,dict(imgsz=2))


if __name__ == '__main__': unittest.main()
