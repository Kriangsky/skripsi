import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

import bisindo_pipeline as b

class PipelineTests(unittest.TestCase):
    def test_two_hand_union_padding_and_clipping(self):
        hands = [[SimpleNamespace(x=.1,y=.2),SimpleNamespace(x=.3,y=.4)],
                 [SimpleNamespace(x=.6,y=.7),SimpleNamespace(x=.9,y=.95)]]
        box = b.union_box(hands,padding=.15)
        np.testing.assert_allclose(box,[0,.0875,1,1])
        self.assertIsNone(b.union_box([]))
        np.testing.assert_allclose(b.yolo_box([0,.2,1,.8]),[.5,.5,1,.6])

    def test_reject_invalid_box(self):
        for box in [[-.1,0,1,1],[0,0,0,1],[0,0,float('nan'),1],[0,.8,1,.2]]:
            with self.assertRaises(ValueError): b.yolo_box(box)

    def fixture(self):
        return pd.DataFrame([dict(video=f'train/{c}/{i}.mp4',source='train',class_name=c)
            for c in b.CLASSES for i in range(300)] + [dict(video=f'test/{c}.mp4',source='test',class_name=c) for c in b.CLASSES])

    def test_video_split_is_deterministic_and_stratified(self):
        fixture=self.fixture()
        a,mode=b.split_rows(fixture)
        z,_=b.split_rows(fixture.sample(frac=1,random_state=1))
        self.assertEqual(a.set_index('video').split.to_dict(),z.set_index('video').split.to_dict())
        self.assertIn('no_signer',mode)
        for c in b.CLASSES:
            self.assertEqual(a[a.class_name.eq(c)].split.value_counts().to_dict(),dict(train=240,val=60,test=1))

    def test_signer_disjoint_globally(self):
        df=self.fixture()
        metadata=df[df.source.eq('train')][['video']].copy()
        metadata['signer']=[f'person_{i%5}' for i in range(len(metadata))]
        out,mode=b.split_rows(df,metadata)
        self.assertIn('signer_disjoint',mode)
        self.assertFalse(set(out[out.split.eq('train')].signer)&set(out[out.split.eq('val')].signer))
        with self.assertRaises(ValueError): b.split_rows(df,metadata.iloc[:1])

    def test_review_binds_box_and_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp); (work/'frames').mkdir()
            image=work/'frames/s.jpg'
            cv2.imwrite(str(image),np.zeros((10,10,3),dtype=np.uint8))
            row=dict(sample_id='s',class_id=0,x1=.1,y1=.1,x2=.8,y2=.8,frame_sha256=b.sha(image),
                review_status='pending',reviewer='',review_note='',review_frame_sha256='',review_box_sha256='')
            pd.DataFrame([row]).to_csv(work/'annotations.csv',index=False)
            b.review_sample(work,'s','tester',box=[.2,.2,.9,.9],note='checked both hands')
            reviewed=pd.read_csv(work/'annotations.csv',keep_default_na=False).iloc[0].to_dict()
            self.assertEqual(reviewed['review_status'],'corrected')
            self.assertEqual(reviewed['review_box_sha256'],b.review_digest(reviewed))
            reviewed['x1']=.3
            self.assertNotEqual(reviewed['review_box_sha256'],b.review_digest(reviewed))
            with self.assertRaises(ValueError): b.review_sample(work,'s','',box=[0,0,1,1])
            with self.assertRaises(ValueError): b.review_sample(work,'s','tester',reject=True)

    def test_training_gate_blocks_before_checkpoint_or_model(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            b.save_json(p/'environment.json',dict(packages={}))
            with patch.object(b,'build_datasets',side_effect=RuntimeError('QC pending')) as build, patch.object(b,'official_checkpoint') as checkpoint:
                with self.assertRaisesRegex(RuntimeError,'QC pending'):
                    b.train_independent(p,p)
                build.assert_called_once()
                checkpoint.assert_not_called()

    def test_release_preserves_shared_samples_and_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); work=root/'work'; (work/'frames').mkdir(parents=True)
            rows=[]
            for cls in b.CLASSES:
                for split in ['train','val','test']:
                    key=f'{cls}_{split}'
                    frame=work/'frames'/f'{key}.jpg'
                    cv2.imwrite(str(frame),np.full((12,16,3),len(rows)*8,dtype=np.uint8))
                    record=dict(sample_id=key,video=f'{key}.mp4',source='test' if split=='test' else 'train',
                        class_name=cls,class_id=b.CLASSES.index(cls),split=split,signer='',video_sha256=key,
                        frame_sha256=b.sha(frame),width=16,height=12,frame_index=1,
                        x1=.1,y1=.2,x2=.8,y2=.9,review_status='approved',reviewer='test',
                        annotation_recipe_sha256=b.annotation_recipe(),review_frame_sha256=b.sha(frame))
                    record['review_box_sha256']=b.review_digest(record)
                    rows.append(record)
            df=pd.DataFrame(rows)
            df.to_csv(work/'annotations.csv',index=False)
            expected=df[['sample_id','video','source','class_name','class_id','split','signer','video_sha256']]
            with patch.object(b,'inventory',return_value=expected):
                self.assertEqual(b.validation_issues(root,work,df),[])
                data=b.build_datasets(root,work)
                for cls in b.BASE:
                    for split in ['train','val','test']:
                        rel=Path('labels')/split/f'{cls}_{split}.txt'
                        self.assertEqual((data/'A'/rel).read_text(),(data/'B'/rel).read_text())
                self.assertEqual(len((data/'A/test_common.txt').read_text().splitlines()),6)
                self.assertEqual((data/'B/labels/test/halo_test.txt').read_text().split()[0],'6')
                changed=df.copy(); changed.loc[0,'x1']=.15
                self.assertTrue(any('box/class changed' in e for e in b.validation_issues(root,work,changed)))

    def test_bulk_review_atomic_validation_and_backup(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);(work/'frames').mkdir()
            rows=[]
            for i in range(2):
                key=f's{i}';image=work/'frames'/f'{key}.jpg'
                cv2.imwrite(str(image),np.full((12,12,3),i*80,dtype=np.uint8))
                rows.append(dict(sample_id=key,class_id=i,x1=.1,y1=.2,x2=.8,y2=.9,
                    frame_sha256=b.sha(image),review_status='pending',reviewer='',review_note='',
                    review_frame_sha256='',review_box_sha256=''))
            pd.DataFrame(rows).to_csv(work/'annotations.csv',index=False)
            initial=(work/'annotations.csv').read_bytes()
            records=[dict(sample_id=r['sample_id'],class_id=r['class_id'],frame_sha256=r['frame_sha256'],
                          box=[.1,.2,.8,.9],note='') for r in rows]
            records[1]['box']=[.2,.3,.9,1.]
            payload=dict(schema=1,kind='approval',all_images_reviewed=True,reviewer='human',
                annotations_sha256=b.sha(work/'annotations.csv'),records=records)
            export=work/'review.json'
            # Invalid last row must not apply even the first valid row.
            records[1]['box']=[0,0,0,0]
            b.save_json(export,payload)
            with self.assertRaises(ValueError):b.import_bulk_review(work,export)
            self.assertEqual((work/'annotations.csv').read_bytes(),initial)
            records[1]['box']=[.2,.3,.9,1.]
            b.save_json(export,payload)
            with patch.object(b,'render_report'):
                result=b.import_bulk_review(work,export)
            self.assertEqual(result.review_status.tolist(),['approved','corrected'])
            self.assertEqual(next(work.glob('annotations_before_bulk_*.csv')).read_bytes(),initial)
            self.assertTrue(all(result.apply(lambda r:r.review_box_sha256==b.review_digest(r),axis=1)))
            with self.assertRaisesRegex(ValueError,'Stale review'):b.import_bulk_review(work,export)
            payload['kind']='draft';b.save_json(export,payload)
            with self.assertRaisesRegex(ValueError,'Approval export required'):b.import_bulk_review(work,export)

if __name__=='__main__':
    unittest.main()
