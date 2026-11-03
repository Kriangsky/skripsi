import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

import bisindo_pipeline as b
import frame_candidates as f

def hand(x1=.2,y1=.2,x2=.5,y2=.6):
    return [SimpleNamespace(x=float(x),y=float(y)) for x,y in zip(np.linspace(x1,x2,21),np.linspace(y1,y2,21))]

class CandidateTests(unittest.TestCase):
    def test_detector_success_only_creates_candidates_not_positive_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            for folder in ['candidate_frames','candidate_records','contact_sheets','frames','candidate_labels']:(work/folder).mkdir()
            video=work/'gesture.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(160,120))
            self.assertTrue(writer.isOpened())
            rng=np.random.default_rng(42)
            for i in range(11):writer.write(rng.integers(0,255,(120,160,3),dtype=np.uint8))
            writer.release()
            detector=SimpleNamespace(detect=lambda image:SimpleNamespace(hand_landmarks=[hand()]))
            record=dict(sample_id='sample',video=video.name,video_sha256=b.sha(video),class_name='saya')
            with patch.object(f._thread,'detector',detector,create=True):
                result=f._process_video(work,work,record)
            self.assertTrue(all(c['quality_status']=='eligible' for c in result['candidates']))
            self.assertEqual([c['frame_index'] for c in result['candidates']],[2,3,4,5,6,7,8])
            self.assertFalse(list((work/'frames').iterdir()))
            self.assertFalse(list((work/'candidate_labels').iterdir()))
            self.assertTrue((work/'contact_sheets/sample.jpg').exists())

    def test_middle_sampling_and_short_videos(self):
        self.assertEqual(f.indices_for(101,f.PRIMARY),[20,30,40,50,60,70,80])
        self.assertEqual(f.indices_for(1,f.PRIMARY),[0])
        self.assertEqual(f.indices_for(0,f.PRIMARY),[])
        self.assertEqual(f.indices_for(2,f.PRIMARY),[0,1])

    def test_hand_roi_blur_border_and_missing_hand_flags(self):
        noise=np.random.default_rng(42).integers(0,255,(240,320,3),dtype=np.uint8)
        box,score,flags=f.assess_hands(noise,[hand()])
        self.assertEqual(flags,[])
        self.assertGreater(score,60)
        blurred=cv2.GaussianBlur(noise,(51,51),15)
        self.assertIn('possible_hand_blur',f.assess_hands(blurred,[hand()])[2])
        self.assertIn('hand_near_border_or_clipped',f.assess_hands(noise,[hand(x1=-.02)])[2])
        self.assertIn('hand_too_small',f.assess_hands(noise,[hand(.2,.2,.205,.205)])[2])
        self.assertEqual(f.assess_hands(noise,[])[2],['no_hands'])
        # A detailed background must not disguise a flat/blurred hand ROI.
        noise[48:145,64:161]=128
        self.assertIn('possible_hand_blur',f.assess_hands(noise,[hand()])[2])

    def fixture(self,work):
        for folder in ['candidate_frames','candidate_labels','frames','overlays']:(work/folder).mkdir()
        rows=[];candidates=[]
        for i in range(2):
            key=f's{i}';cid=f'{key}_f00000030';image=work/'candidate_frames'/f'{cid}.jpg'
            cv2.imwrite(str(image),np.full((120,160,3),i*100,dtype=np.uint8))
            video=work/f'video{i}.mp4';video.write_bytes(bytes([i]))
            rows.append(dict(sample_id=key,video=video.name,source='train',split='train' if i==0 else 'val',
                signer='',class_id=i,class_name=b.CLASSES[i],video_sha256=b.sha(video),annotation_recipe_sha256=b.annotation_recipe(),
                frame_index=-1,selected_candidate_id='',frame_sha256='',width=0,height=0,num_hands=0,x1='',y1='',x2='',y2='',
                auto_status='needs_review',review_status='needs_review',frame_review_status='needs_review',reviewer='',
                review_note='',review_frame_sha256='',review_box_sha256='',review_selection_sha256='',
                quality_override_reason='',rejection_reason='gesture_not_yet_verified',rejected_candidates='{}',recommended_candidate_id=cid,candidate_count=1))
            candidates.append(dict(sample_id=key,candidate_id=cid,frame_index=30,frame_count=101,
                frame_path=image.relative_to(work).as_posix(),frame_sha256=b.sha(image),width=160,height=120,num_hands=1,
                box=json.dumps([.1,.2,.8,.9]),focus=100.,quality_status='eligible',quality_flags='[]',landmarks='[]'))
        pd.DataFrame(rows).to_csv(work/'annotations.csv',index=False)
        pd.DataFrame(candidates).to_csv(work/'candidates.csv',index=False)
        b.save_json(work/'candidate_config.json',dict(protocol='human_selected_positive_v1'))
        payload=dict(schema=1,kind='frame_selection',reviewer='human',annotations_sha256=b.sha(work/'annotations.csv'),
            candidates_sha256=b.sha(work/'candidates.csv'),records=[dict(sample_id=r['sample_id'],class_id=r['class_id'],
                candidate_id=c['candidate_id'],box=[.1,.2,.8,.9],gesture_verified=True,rejected_candidates={}) for r,c in zip(rows,candidates)])
        return pd.DataFrame(rows),payload

    def test_no_positive_from_detection_and_partial_human_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);before,payload=self.fixture(work)
            self.assertEqual(len(f.selection_issues(work,before)),2)
            self.assertFalse(list((work/'frames').iterdir()))
            # Reviewer rejects the other video's preparation pose instead of forcing a label.
            other=payload['records'][1]
            other['rejected_candidates']={other['candidate_id']:'preparation_pose'}
            other.update(candidate_id=None,reason='all candidates in preparation pose',note='find another frame',request_more=False)
            export=work/'review.json';b.save_json(export,payload)
            with patch.object(f,'render_report'):
                result=f.import_review(work,work,export)
            self.assertEqual(result.frame_review_status.tolist(),['approved','needs_review'])
            self.assertEqual(result.frame_index.tolist(),[30,-1])
            self.assertTrue(result[['class_name','class_id','split','source']].equals(before[['class_name','class_id','split','source']]))
            self.assertEqual(len(list((work/'frames').glob('*.jpg'))),1)
            self.assertIn('preparation_pose',result.iloc[1].rejected_candidates)
            self.assertEqual(len(f.selection_issues(work,result)),1)
            result.loc[0,'frame_index']=40
            self.assertTrue(any('mismatch' in e for e in f.selection_issues(work,result)))

    def test_atomic_reject_without_semantic_review_or_with_stale_catalog(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);_,payload=self.fixture(work)
            original=(work/'annotations.csv').read_bytes()
            payload['records'][1]['gesture_verified']=False
            export=work/'review.json';b.save_json(export,payload)
            with self.assertRaisesRegex(ValueError,'Explicit verification'):f.import_review(work,work,export)
            self.assertEqual(original,(work/'annotations.csv').read_bytes())
            self.assertFalse(list((work/'frames').iterdir()))
            payload['candidates_sha256']='stale';b.save_json(export,payload)
            with self.assertRaisesRegex(ValueError,'Stale frame review'):f.import_review(work,work,export)

    def test_flagged_candidate_requires_justification(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);_,payload=self.fixture(work)
            candidates=pd.read_csv(work/'candidates.csv');candidates.loc[0,'quality_status']='flagged'
            candidates.to_csv(work/'candidates.csv',index=False)
            payload['candidates_sha256']=b.sha(work/'candidates.csv')
            export=work/'review.json';b.save_json(export,payload)
            with self.assertRaisesRegex(ValueError,'justification'):f.import_review(work,work,export)

if __name__=='__main__':unittest.main()
