"""exploratory/development regression tests; synthetic tensors only, no model fits."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
# Load before legacy sys.modules mocks: reimporting Torch extensions cannot undo C++ registrations.
import torchvision
import classification_research as r


class DevelopmentTests(unittest.TestCase):
    def test_stage1b_gate_needs_scoped_approval_but_no_new_addendum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); settings=r.Settings()
            (root/'PREREGISTRATION.md').write_text('candidate')
            approval=root/'approval.json'
            record=dict(kind='bisindo_training_approval',scope='stage1b',approved_by='researcher',
                work_names=['new'],training=asdict(settings),
                protocol_sha256={'PREREGISTRATION.md':r.base.b.sha(root/'PREREGISTRATION.md')})
            approval.write_text(json.dumps(record))
            self.assertEqual(r.authorize_training(root,root/'new',settings,approval),r.base.b.sha(approval))
            record['scope']='stage2';approval.write_text(json.dumps(record))
            with self.assertRaises(PermissionError):r.authorize_training(root,root/'new',settings,approval)

    def test_mask_union_padding_missing_and_invalid(self):
        import classification_diagnostics as d
        image=np.zeros((100,200,3),np.uint8)
        hand=[[.25,.25],[.5,.5]]+[[.3,.3]]*19
        result,mask,record=d.mask_hands(image,[hand])
        self.assertEqual(record['stratum'],'landmarks_available')
        self.assertEqual(mask.shape,(100,200))
        self.assertTrue(np.all(result[mask>0]==114))
        self.assertTrue(np.all(result[mask==0]==0))
        self.assertEqual(record['roi_xyxy'],[42,21,108,54])
        same,empty,record=d.mask_hands(image,[])
        np.testing.assert_array_equal(same,image)
        self.assertEqual(empty.sum(),0)
        self.assertEqual(record['stratum'],'landmarks_unavailable')
        with self.assertRaises(ValueError):d.mask_hands(image,[[[float('nan'),.2]]*21])

    def test_measure_train_preserves_weights_modes_and_rng(self):
        import torch
        import classification_diagnostics as d
        from torch.utils.data import DataLoader,TensorDataset
        model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.BatchNorm1d(4),torch.nn.Dropout(.5),torch.nn.Linear(4,6))
        model.train(); model[1].eval()
        weights={k:v.clone() for k,v in model.state_dict().items()}
        modes=[m.training for m in model.modules()];rng=torch.random.get_rng_state().clone()
        loader=DataLoader(TensorDataset(torch.ones(4,1,2,2),torch.zeros(4,dtype=torch.long)),batch_size=2,
                          generator=torch.Generator().manual_seed(1))
        record=d.measure_classifier(model,loader,'cpu')
        self.assertEqual(record['n'],4)
        self.assertTrue(0<=record['top1']<=1)
        self.assertEqual(modes,[m.training for m in model.modules()])
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,weights[k]))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))

    def test_mask_release_saves_unavailable_and_rejects_overwrite(self):
        from PIL import Image
        import pandas as pd
        import classification_diagnostics as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'run';source=root/'cache';source.mkdir()
            (work/'selected_frames').mkdir(parents=True)
            raw=work/'selected_frames/s1.jpg';Image.fromarray(np.zeros((10,20,3),np.uint8)).save(raw)
            digest=r.base.b.sha(raw)
            pd.DataFrame([dict(sample_id='s1',video='train/s1.mov',video_sha256='video',split='val',
                class_id=0,class_name='saya',candidate_id='c1',source_frame_sha256=digest)]).to_csv(work/'manifest.csv',index=False)
            pd.DataFrame([dict(sample_id='s1',candidate_id='c1',frame_sha256=digest,landmarks='[]')]).to_csv(source/'candidates.csv',index=False)
            frame=d.prepare_masks(work,source,root/'exploratory_mask')
            self.assertEqual(frame.stratum.tolist(),['landmarks_unavailable'])
            self.assertEqual(frame.coverage.tolist(),[0])
            self.assertTrue((root/'exploratory_mask/exploratory_masks/s1.png').is_file())
            with Image.open(root/'exploratory_mask/exploratory_images/s1.png') as opened:
                self.assertEqual(opened.size,(224,224))
            with self.assertRaises(FileExistsError):d.prepare_masks(work,source,root/'exploratory_mask')
            with self.assertRaises(ValueError):d.fresh_output(work,root/'experiment_yolo26_v3/new')
            from unittest.mock import patch
            frame.loc[0,'split']='train';frame.to_csv(root/'exploratory_mask/exploratory_mask_coverage.csv',index=False)
            with patch.object(r,'load_settings',return_value=r.Settings()),patch.object(r,'validate_release'):
                with self.assertRaisesRegex(ValueError,'provenance'):
                    d.evaluate_masked(work,root/'exploratory_mask',root/'eval')

    def test_training_and_probe_dispatch_default_do_nothing(self):
        import classification_workflow as workflow
        from unittest.mock import patch
        with patch.object(r,'train') as train:
            self.assertIn('SKIPPED',workflow.train_stage1b(Path.cwd(),'exploratory_test',{}))
            train.assert_not_called()
        with patch.object(r,'load_settings',return_value=r.Settings()),patch.object(r,'authorize_training',side_effect=PermissionError):
            with self.assertRaises(PermissionError):workflow.train_stage1b(Path.cwd(),'exploratory_test',{},True)

    def test_smoke_uses_parameterized_release_validator(self):
        import classification_workflow as workflow
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'research_config.json').write_text('{}')
            with patch.object(r,'load_settings',return_value=r.Settings(44,15,448)),patch.object(r,'validate_release',side_effect=ValueError('parameterized validation')) as validate:
                with self.assertRaisesRegex(ValueError,'parameterized validation'):workflow.inference_smoke(root,root.parent/'unused')
                validate.assert_called_once_with(root,r.Settings(44,15,448))

    def test_linear_backbone_eval_features_and_no_approval_no_fit(self):
        import torch
        import classification_diagnostics as d
        from torch.utils.data import DataLoader,TensorDataset
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.drop=torch.nn.Dropout(.8);self.linear=torch.nn.Linear(4,8)
            def forward(self,x):return self.linear(self.drop(x.flatten(1)))
        model=Encoder()
        loader=DataLoader(TensorDataset(torch.ones(4,1,2,2),torch.arange(4)),batch_size=2)
        features,labels=d.extract_features(model,model.linear,loader,'cpu')
        np.testing.assert_allclose(features.numpy(),1)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertEqual(labels.tolist(),[0,1,2,3])
        with self.assertRaises(PermissionError):d.fit_linear_probe('missing','A','missing',approval=None)

    def test_frozen_feature_extraction_does_not_change_backbone(self):
        import torch
        import classification_diagnostics as d
        from torch.utils.data import DataLoader,TensorDataset
        model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.BatchNorm1d(4),torch.nn.Linear(4,8))
        original={key:value.clone() for key,value in model.state_dict().items()}
        loader=DataLoader(TensorDataset(torch.ones(6,1,2,2),torch.arange(6)),batch_size=2,
                          generator=torch.Generator().manual_seed(1))
        features,_=d.extract_features(model,model[-1],loader,'cpu')
        self.assertEqual(features.shape,(6,4))
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,original[key]))

    def test_train_curve_records_each_epoch_once_without_touching_weights(self):
        import torch
        from types import SimpleNamespace
        from PIL import Image
        import pandas as pd
        import classification_diagnostics as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'train'
            for name in r.base.folder_names(6):
                path=folder/name;path.mkdir(parents=True)
                Image.fromarray(np.zeros((2,2,3),np.uint8)).save(path/'s.png')
            model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(12,6)).train()
            before={key:value.clone() for key,value in model.state_dict().items()}
            trainer=SimpleNamespace(model=model,save_dir=root,device='cpu',epoch=0)
            d.log_train_epoch(trainer,folder,2,'A');trainer.epoch=1
            d.log_train_epoch(trainer,folder,2,'A')
            frame=pd.read_csv(root/'exploratory_train_curve.csv')
            self.assertEqual(frame.epoch.tolist(),[1,2]);self.assertEqual(frame.n.tolist(),[6,6])
            for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,before[key]))
            with self.assertRaises(ValueError):d.log_train_epoch(trainer,folder,2,'A')

    def test_planning_grid_margins_and_cv_are_explicit(self):
        import power_precision as p
        frame=p.development_scenarios(repetitions=30)
        self.assertEqual(set(frame.ni_margin),{.02,.03,.05})
        self.assertEqual(set(frame.n_common6),{300,360,1800})
        self.assertTrue(frame.method.str.contains('NOT exact').all())
        subset=frame[(frame.n_common6==300)&(frame.correctness_discordance_assumed==.05)&(frame.paired_seed_sd_assumed==.01)]
        self.assertTrue(subset.sort_values('ni_margin').ni_pass_probability.is_monotonic_increasing)

    def test_notebook_has_module_calls_and_training_false(self):
        path=Path(__file__).with_name('BISINDO_Classification_Vocabulary_Expansion.ipynb')
        self.assertTrue(path.is_file())
        nb=json.loads(path.read_text(encoding='utf8'))
        code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        self.assertIn('RUN_TRAINING = False',code)
        self.assertIn('exploratory',code)
        self.assertIn('workflow.train_stage1b',code)
        self.assertIn('workflow.diagnose_stage1b',code)
        for c in nb['cells']:
            if c['cell_type']=='code':
                compile(''.join(c['source']),str(path),'exec')
                self.assertIsNone(c['execution_count'])
                self.assertEqual(c['outputs'],[])


if __name__=='__main__':unittest.main()
