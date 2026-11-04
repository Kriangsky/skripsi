from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import classification_pilot as legacy
import classification_research as r


class ResearchClassificationTests(unittest.TestCase):
    def test_settings_validation_and_legacy_defaults_preserved(self):
        original=dict(legacy.CONFIG)
        core=r.configured_core(r.Settings(43,15,320))
        self.assertEqual(core.CONFIG['seed'],43)
        self.assertEqual(core.CONFIG['epochs'],15)
        self.assertEqual(core.CONFIG['imgsz'],320)
        self.assertEqual(core.letterbox(np.zeros((20,40,3),np.uint8)).shape,(320,320,3))
        self.assertEqual(legacy.SIZE,224)
        self.assertEqual(legacy.CONFIG,original)
        self.assertEqual(core.sample_videos.__code__.co_code,legacy.sample_videos.__code__.co_code)
        for args in [(41,30,224),(42,0,224),(42,30,225)]:
            with self.assertRaises(ValueError):r.Settings(*args)

    def test_work_rejects_existing_source_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(r.new_work(root,root/'new'),root/'new')
            for work in [root,root/'train',root.parent/'escape']:
                with self.assertRaises(ValueError):r.new_work(root,work)
            (root/'old').mkdir()
            with self.assertRaises(ValueError):r.new_work(root,root/'old')

    def test_locked_settings_hash_source_and_cli_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);settings=r.Settings(44,10,448)
            record=r.settings_record(settings);record['settings_fingerprint']=legacy.frames.digest(record)
            legacy.dump(work/'research_config.json',record)
            self.assertEqual(r.load_settings(work),settings)
            with self.assertRaises(ValueError):r.load_settings(work,{'imgsz':224})
            record['training']['seed']=45;legacy.dump(work/'research_config.json',record)
            with self.assertRaises(ValueError):r.load_settings(work)

    def test_training_requires_explicit_approval_before_gpu_or_fit(self):
        settings=r.Settings()
        with patch.object(r,'load_settings',return_value=settings),patch.object(r,'configured_core') as factory:
            with self.assertRaises(PermissionError):r.train(Path.cwd(),Path.cwd(),Path.cwd()/'new')
            factory.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            approval=Path(tmp)/'approval.json';approval.write_text(json.dumps({'kind':'bisindo_training_approval','scope':'stage1b','approved_by':'researcher','work_names':['new'],'training':asdict(settings)}))
            with self.assertRaises(PermissionError):r.authorize_training(tmp,Path(tmp)/'new',settings,approval)

    def test_evaluation_rejects_existing_or_wrong_epoch_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);(work/'evaluation').mkdir();(work/'evaluation/evaluation_started.json').write_text('{}')
            with patch.object(r,'load_settings',return_value=r.Settings(42,15,320)),patch.object(r,'configured_core') as factory:
                with self.assertRaises(ValueError):r.evaluate(work)
                factory.assert_not_called()
            (work/'evaluation/evaluation_started.json').unlink();(work/'runs').mkdir()
            legacy.dump(work/'runs/trained_models.json',dict(models=[dict(epoch=30),dict(epoch=30)]))
            with patch.object(r,'load_settings',return_value=r.Settings(42,15,320)),patch.object(r,'validate_release'):
                with self.assertRaises(ValueError):r.evaluate(work)

    def test_cli_flags_implemented_and_work_alias(self):
        with patch.object(r,'prepare') as prepare:
            r.main(['prepare','--work-dir','new','--seed','45','--epochs','15','--imgsz','448'])
            self.assertEqual(prepare.call_args.args[3],r.Settings(45,15,448))
            self.assertEqual(prepare.call_args.args[2],Path('new'))
        with patch.object(r,'evaluate') as evaluate:
            r.main(['evaluate','--work','new','--seed','45'])
            self.assertEqual(evaluate.call_args.args[1]['seed'],45)

    def test_config_fingerprint_changes_with_each_parameter(self):
        baseline=legacy.frames.digest(r.settings_record(r.Settings()))
        for settings in [r.Settings(43),r.Settings(42,15),r.Settings(42,30,320)]:
            self.assertNotEqual(baseline,legacy.frames.digest(r.settings_record(settings)))

    def test_prepare_propagates_size_seed_epoch_and_release_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'new';checkpoint=root/'yolo26n-cls.pt';checkpoint.write_bytes(b'fixture')
            settings=r.Settings(43,15,320);core=r.configured_core(settings)
            original_sha=legacy.b.sha

            def fake_prepare(root,source,work):
                (work/'reports').mkdir();(work/'manifest.csv').write_text('fixture')
                legacy.dump(work/'reports/experiment_config.json',dict(training=core.CONFIG))
                legacy.dump(work/'release.json',dict(files={},source_snapshot={}))
                return {'ready':True}

            core.prepare=fake_prepare
            with patch.object(r,'configured_core',return_value=core),patch.object(legacy.b,'sha',side_effect=lambda p:r.PINNED_SHA if Path(p)==checkpoint else original_sha(p)):
                r.prepare(root,root/'source',work,settings,checkpoint)
            record=legacy.read_json(work/'reports/experiment_config.json')
            self.assertEqual(record['training']['imgsz'],320)
            self.assertEqual(record['training']['seed'],43)
            self.assertIn('15 epochs',record['evaluation_checkpoint'])
            self.assertIn('320x320',record['preprocessing'])
            release=legacy.read_json(work/'release.json')
            self.assertEqual(release['research_config_sha256'],original_sha(work/'research_config.json'))
            self.assertEqual(release['fingerprint'],legacy.frames.digest(dict(manifest=original_sha(work/'manifest.csv'),config=record)))
            self.assertEqual(r.load_settings(work),settings)

    def test_mocked_independent_fits_forward_parameters_and_fixed_last_epoch(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'new';(work/'reports').mkdir(parents=True);(work/'runs').mkdir()
            legacy.dump(work/'reports/readiness.json',dict(ready=True))
            initial=root/'yolo26n-cls.pt';initial.write_bytes(b'fixture')
            settings=r.Settings(44,15,320);core=r.configured_core(settings);calls=[]

            class FakeYOLO:
                def __init__(self,path):
                    self.task='classify';self.names=dict(enumerate(range(1000)));calls.append({'initial':path})
                def add_callback(self,*args):pass
                def train(self,**kwargs):
                    calls[-1]['args']=kwargs
                    folder=work/'runs'/kwargs['name'];(folder/'weights').mkdir(parents=True)
                    (folder/'weights/last.pt').write_bytes(b'fixture last')
                    pd.DataFrame({'epoch':range(1,settings.epochs+1)}).to_csv(folder/'results.csv',index=False)
                    self.trainer=SimpleNamespace(save_dir=folder,actual_steps=15,args=SimpleNamespace(**{k:v for k,v in kwargs.items() if k!='trainer'}))

            core.probe=lambda _:dict(ready=True,checkpoint=dict(path=str(initial),sha256=r.PINNED_SHA))
            original_sha=legacy.b.sha
            with (patch.object(r,'load_settings',return_value=settings),patch.object(r,'authorize_training',return_value='fixture approval'),
                  patch.object(r,'validate_release',return_value=dict(fingerprint='fixture',source_snapshot={})),
                  patch.object(r,'configured_core',return_value=core),patch.object(core.detection,'local_yolo',return_value=FakeYOLO),
                  patch.object(legacy.b,'sha',side_effect=lambda p:r.PINNED_SHA if Path(p)==initial else original_sha(p)),
                  patch.dict('sys.modules',{'ultralytics.models.yolo.classify':SimpleNamespace(ClassificationTrainer=object)}),
                  patch.object(torch.cuda,'empty_cache'),patch.object(r,'evaluate') as evaluate):
                r.train(root,root/'source',work,approval=root/'approved.json')
            self.assertEqual([call['initial'] for call in calls],[str(initial),str(initial)])
            for call in calls:
                for key,value in asdict(settings).items():self.assertEqual(call['args'][key],value)
            record=legacy.read_json(work/'runs/trained_models.json')
            self.assertEqual({row['epoch'] for row in record['models']},{15})
            self.assertTrue(all(row['path'].endswith('last.pt') for row in record['models']))
            evaluate.assert_called_once_with(work)


if __name__=='__main__':unittest.main()
