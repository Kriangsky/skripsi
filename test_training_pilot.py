import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

import training_pilot as t
from training_pilot_eval import match_frame


def inventory():
    return pd.DataFrame([dict(sample_id=f'{n}-{split}-{i}', video=f'train/{cls}/{split}{i}.mp4',
        source='train', split=split, class_name=cls, class_id=n, signer='',
        video_sha256=f'{n}{split}{i}') for n, cls in enumerate(t.CLASSES)
        for split, count in [('train', 65), ('val', 25)] for i in range(count)])


class TrainingPilotTests(unittest.TestCase):
    def test_sampling_balanced_shared_mapping_reproducible(self):
        data = inventory()
        reviews = {r.sample_id:dict(status='positive') for r in data.itertuples()}
        a, counts, excluded, errors = t.select_samples(data, reviews, set())
        z, _, _, _ = t.select_samples(data.sample(frac=1, random_state=9), reviews, set())
        self.assertFalse(errors)
        self.assertEqual(a.sample_id.tolist(), z.sample_id.tolist())
        self.assertEqual(len(a), 640)
        self.assertTrue(a.groupby(['class_name','split']).size().unstack()['train'].eq(60).all())
        self.assertEqual(a[a.class_name.isin(t.BASE)].groupby('split').size().to_dict(),dict(train=360,val=120))
        self.assertEqual(len(excluded), 80)

    def test_no_external_or_human_development_samples_in_validation(self):
        data = inventory()
        external = data.iloc[:1].assign(sample_id='external', source='test', split='test', video='test/external.mp4', video_sha256='external-content')
        data = pd.concat([data, external], ignore_index=True)
        reviews = {r.sample_id:dict(status='positive') for r in data.itertuples()}
        human = set(data[data.split.eq('val')].sample_id.iloc[:10])
        selected, _, excluded, errors = t.select_samples(data, reviews, human)
        self.assertTrue(selected.source.eq('train').all())
        self.assertFalse(set(selected[selected.split.eq('val')].sample_id) & human)
        self.assertTrue(errors)
        self.assertIn('human_pilot_development', set(excluded.reason))

    def test_no_oversampling_and_preserve_unreviewed_and_rejected(self):
        data = inventory()
        reviews = {data.iloc[0].sample_id:dict(status='excluded', reason='temporal_ambiguity')}
        selected, counts, excluded, errors = t.select_samples(data, reviews, set())
        self.assertEqual(len(selected),0)
        self.assertEqual(len(excluded),len(data))
        self.assertIn('temporal_ambiguity',set(excluded.reason))
        self.assertEqual(len(errors),16)
        self.assertTrue(counts.available_positive.eq(0).all())

    def test_signer_and_session_independence_fail_closed(self):
        selected=inventory().groupby(['class_name','split']).head(1)
        self.assertTrue(t.metadata_issues(selected,pd.DataFrame(columns=['video','signer','session'])))
        metadata=selected[['video','split']].copy()
        metadata['signer']=metadata['split'].map({'train':'person1','val':'person2'})
        metadata['session']=metadata['split'].map({'train':'session1','val':'session2'})
        self.assertFalse(t.metadata_issues(selected,metadata))
        metadata['signer']='same'
        self.assertTrue(any('signer' in x for x in t.metadata_issues(selected,metadata)))
        metadata['signer']=metadata['split']
        metadata['session']='same'
        self.assertTrue(any('session' in x for x in t.metadata_issues(selected,metadata)))

    def test_label_conversion_and_bounds(self):
        self.assertEqual(t.label_line(5,[.1,.2,.9,.8]).split()[0],'5')
        self.assertEqual([float(x) for x in t.label_line(5,[.1,.2,.9,.8]).split()[1:]],[.5,.5,.8,.6])
        for cls,box in [(8,[0,0,1,1]),(0,[0,0,0,1]),(0,[-.1,0,1,1]),(0,[.5,.5,.500000000001,.6])]:
            with self.assertRaises(ValueError):t.label_line(cls,box)

    def test_extra_class_predictions_kept_in_common6_matching(self):
        result=match_frame(0,[.1,.1,.9,.9],[dict(class_id=6,confidence=.9,box=[.1,.1,.9,.9])],8)
        self.assertEqual((result['tp'],result['fp'],result['fn']),(0,1,1))
        self.assertEqual(result['confusion_entries'],[(0,6)])
        self.assertEqual(result['predictions'][0]['class_id'],6)

    def test_matching_duplicates_iou_threshold_and_confidence(self):
        preds=[dict(class_id=0,confidence=.9,box=[.1,.1,.9,.9]),
               dict(class_id=7,confidence=.8,box=[.1,.1,.9,.9]),
               dict(class_id=0,confidence=.2,box=[.1,.1,.9,.9])]
        result=match_frame(0,[.1,.1,.9,.9],preds,8)
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,1,0))
        self.assertEqual(result['confusion_entries'],[(0,0),(8,7)])
        result=match_frame(0,[.1,.1,.9,.9],[dict(class_id=0,confidence=.9,box=[0,0,.05,.05])],8)
        self.assertEqual((result['tp'],result['fp'],result['fn']),(0,1,1))
        self.assertIn((0,8),result['confusion_entries'])
        self.assertEqual(match_frame(0,[.1,.1,.9,.9],[],8)['fn'],1)

    def test_training_gate_precedes_model_and_never_falls_back_to_cpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(t,'prepare',return_value=dict(ready=False,blockers=['signer metadata missing'])), patch.object(t,'local_yolo') as factory:
                with self.assertRaisesRegex(RuntimeError,'signer metadata missing'):
                    t.train(root,root/'source',root/'work')
                factory.assert_not_called()

    def test_review_attestations_required_for_every_positive(self):
        row=dict(candidate_id='c',gesture_verified=True,box_verified=True,
            all_required_hands_visible=True,box=[.1,.1,.9,.9],quality_override_reason='',semantic_status='positive')
        candidate=dict(candidate_id='c',sample_id='s',quality_flags=[],quality_status='eligible')
        t.validate_positive(row,candidate)
        for key in ['gesture_verified','box_verified','all_required_hands_visible']:
            with self.assertRaises(ValueError):t.validate_positive(dict(row,**{key:False}),candidate)
        with self.assertRaises(ValueError):t.validate_positive(dict(row,reason='transition_pose'),candidate)
        with self.assertRaises(ValueError):t.validate_positive(row,dict(candidate,quality_flags=['possible_hand_blur'],quality_status='flagged'))

    def fixture(self, root):
        source, work = root/'v3', root/'controlled'
        for path in [source/'candidate_frames',source/'candidate_records',source/'contact_sheets',work/'reports',work/'review',work/'datasets',work/'runs']:
            path.mkdir(parents=True,exist_ok=True)
        data=inventory()
        rng=np.random.default_rng(42)
        catalog={}
        reviews={}
        for i,row in enumerate(data.to_dict('records')):
            video=root/row['video'];video.parent.mkdir(parents=True,exist_ok=True);video.write_bytes(str(i).encode())
            data.loc[i,'video_sha256']=t.b.sha(video)
            cid=row['sample_id']+'_f00000001'
            frame=source/'candidate_frames'/f'{cid}.jpg'
            cv2.imwrite(str(frame),rng.integers(0,255,(120,160,3),dtype=np.uint8))
            c=dict(sample_id=row['sample_id'],candidate_id=cid,frame_index=1,frame_count=10,
                frame_path=frame.relative_to(source).as_posix(),frame_sha256=t.b.sha(frame),width=160,height=120,
                num_hands=1,box=[.1,.1,.9,.9],focus=100.,quality_status='eligible',quality_flags=[],
                landmarks=[[[float(x),float(y)] for x,y in zip(np.linspace(.2,.6,21),np.linspace(.2,.7,21))]])
            catalog[cid]=c
            cache=dict(sample_id=row['sample_id'],video=row['video'],video_sha256=data.loc[i,'video_sha256'],
                recipe='fixture',candidates=[{k:v for k,v in c.items() if k!='sample_id'}])
            t.dump(source/'candidate_records'/f"{row['sample_id']}.json",cache)
            reviews[row['sample_id']]=dict(status='positive',candidate_id=cid,gesture_verified=True,box_verified=True,
                all_required_hands_visible=True,semantic_status='positive',reason='',box=c['box'],
                frame_sha256=c['frame_sha256'],reviewer='fixture-reviewer',timestamp='2026-11-01T01:00:00Z',quality_override_reason='')
        data.to_csv(source/'split_lock.csv',index=False)
        data.to_csv(source/'annotations.csv',index=False)
        pd.DataFrame([{k:json.dumps(v) if isinstance(v,list) else v for k,v in c.items()} for c in catalog.values()]).to_csv(source/'candidates.csv',index=False)
        t.dump(source/'candidate_config.json',dict(split_lock_sha256=t.b.sha(source/'split_lock.csv'),quality_heuristics=t.f.QUALITY,annotation_recipe_sha256='fixture'))
        t.dump(source/'checkpoint.json',dict(filename='fixture.pt'))
        (root/'experiment_yolo26_pilot').mkdir()
        pd.DataFrame(columns=['video_id','video_sha256']).to_csv(root/'experiment_yolo26_pilot/pilot_manifest_private.csv',index=False)
        metadata=data[['video','split']].assign(signer=lambda d:'person-'+d.split,session=lambda d:'session-'+d.split)
        return source,work,data,catalog,reviews,metadata

    def test_release_yaml_shared_bytes_label_integrity_and_contact_sheets(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,work,data,catalog,reviews,metadata=self.fixture(Path(tmp))
            selected,_,_,errors=t.select_samples(data,reviews,set())
            self.assertFalse(errors)
            release=t.release_datasets(work,source,selected,reviews,catalog,metadata,'fixture-metadata')
            self.assertEqual(len(release['contact_sheets']),32)
            t.validate_release(work)
            self.assertEqual(len(list((work/'datasets/A/images/val').glob('*.jpg'))),120)
            self.assertEqual(len(list((work/'datasets/B/images/val').glob('*.jpg'))),160)
            common=t.yaml.safe_load((work/'datasets/B/common6.yaml').read_text())
            self.assertEqual(common['names'],t.CLASSES)
            self.assertEqual(len((work/'datasets/B/common6_val.txt').read_text().splitlines()),120)
            row=selected.iloc[0]
            relative=Path('labels')/row.split/f'{row.sample_id}.txt'
            self.assertEqual((work/'datasets/A'/relative).read_bytes(),(work/'datasets/B'/relative).read_bytes())
            (work/'datasets/A'/relative).write_text('7 .5 .5 .5 .5\n')
            with self.assertRaises(ValueError):t.validate_release(work)

    def test_prepare_import_snapshot_preservation_and_deterministic_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,work,data,catalog,reviews,metadata=self.fixture(root)
            before=t.source_snapshot(source)
            with patch.object(t,'probe_environment',return_value=dict(ready=True,blockers=[])):
                state=t.prepare(root,source,work)
                self.assertFalse(state['ready'])
                self.assertEqual(state['queued_review_n'],640)
                self.assertEqual(state['selected_n'],0)
                queue=t.csv(work/'review_queue.csv')
                html=(work/'review/frame_review.html').read_text(encoding='utf-8')
                self.assertIn('box_verified:true,all_required_hands_visible:true',html)
                self.assertNotIn("$('bulk').onclick=",html)
                self.assertNotIn("$('video').src='../'",html)
                self.assertIn('temporal_ambiguity',html)
                records=[dict(sample_id=r.sample_id,class_id=int(r.class_id),candidate_id=None,
                    reason='gesture_not_yet_verified',note='',request_more=False,rejected_candidates={}) for r in queue.itertuples()]
                records[0].update({k:v for k,v in reviews[records[0]['sample_id']].items() if k!='status'})
                records[1].update(reason='temporal_ambiguity',semantic_status='excluded',note='No single representative phase')
                payload=dict(schema=1,kind='training_pilot_review',reviewer='fixture-reviewer',
                    annotations_sha256=t.b.sha(work/'review_queue.csv'),candidates_sha256=t.b.sha(source/'candidates.csv'),records=records)
                review_file=root/'review.json';t.dump(review_file,payload)
                t.import_review(root,source,work,review_file)
                with self.assertRaises(ValueError):t.import_review(root,source,work,review_file)
                updated=t.read_json(work/'reviews.json')
                self.assertEqual(sum(r['status']=='positive' for r in updated.values()),1)
                self.assertEqual(sum(r['status']=='excluded' for r in updated.values()),1)
                t.prepare(root,source,work)
                next_queue=t.csv(work/'review_queue.csv')
                self.assertEqual(len(next_queue),640)
                self.assertNotIn(records[1]['sample_id'],set(next_queue.sample_id))
                self.assertIn(records[0]['sample_id'],set(next_queue.sample_id))
            self.assertEqual(before,t.source_snapshot(source))

    def test_training_two_fresh_coco_models_identical_controls_no_sequential_weights(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';(work/'runs').mkdir(parents=True)
            initial=root/'yolo26n.pt';initial.write_bytes(b'fixture COCO')
            calls=[]
            class FakeYOLO:
                def __init__(self,path):
                    calls.append(dict(initial=path));self.names={i:str(i) for i in range(80)}
                    self.model=SimpleNamespace(end2end=False,yaml=dict(end2end=True),model=[SimpleNamespace(one2one_cv2=[])])
                def add_callback(self,*args):pass
                def train(self,**kwargs):
                    calls[-1]['train']=kwargs
                    folder=work/'runs'/kwargs['name'];(folder/'weights').mkdir(parents=True)
                    (folder/'weights/best.pt').write_bytes(kwargs['name'].encode())
                    self.trainer=SimpleNamespace(save_dir=folder,epoch=29,
                        actual_steps=690 if kwargs['name']=='model_A' else 900,args=SimpleNamespace(batch=16,epochs=30))
            with (patch.object(t,'prepare',return_value=dict(ready=True)),patch.object(t,'validate_release',return_value=dict(fingerprint='fixture')),
                 patch.object(t,'checkpoint_info',return_value=dict(path=str(initial),sha256=t.b.sha(initial))),patch.object(t,'local_yolo',return_value=FakeYOLO),
                 patch.object(t.b,'record_environment'),patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'empty_cache'),
                 patch.dict('sys.modules',{'ultralytics.models.yolo.detect':SimpleNamespace(DetectionTrainer=object)}),patch('training_pilot_eval.evaluate') as evaluate):
                t.train(root,root/'source',work)
            self.assertEqual([r['initial'] for r in calls],[str(initial),str(initial)])
            for key,value in t.PILOT_CONFIG.items():
                self.assertEqual(calls[0]['train'][key],value)
                self.assertEqual(calls[1]['train'][key],value)
            self.assertNotEqual(calls[0]['train']['data'],calls[1]['train']['data'])
            self.assertTrue((work/'runs/trained_models.json').exists())
            evaluate.assert_called_once_with(work)


if __name__=='__main__':unittest.main()
