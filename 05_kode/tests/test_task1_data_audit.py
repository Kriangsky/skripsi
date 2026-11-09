"""Synthetic tests for read-only data audit validation and duplicate scope."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

import task1_data_audit as audit


class DataAuditTests(unittest.TestCase):
    def frame(self):
        video='train/saya/sample.mp4'
        return pd.DataFrame([dict(sample_id=audit.pipeline.sid(video),video=video,
                                 source='train',split='val',class_id=0,class_name='saya',
                                 video_sha256='a'*64,signer='')])

    def test_identity_rejects_gt_split_path_and_duplicate_errors(self):
        audit.validate_identity(self.frame())
        for field,value in [('class_id',7),('split','test'),('source','test'),
                            ('video','train/../sample.mp4'),('sample_id','wrong'),('video_sha256','x')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                changed=self.frame();changed.loc[0,field]=value
                audit.validate_identity(changed)
        with self.assertRaises(ValueError):
            audit.validate_identity(pd.concat([self.frame(),self.frame()]))

    def test_same_video_reuse_is_not_cross_video_leakage(self):
        rows=[dict(sample_id='a',split='train',sha='same'),dict(sample_id='a',split='train',sha='same')]
        self.assertEqual(audit.duplicate_groups(rows,'sha'),[])
        rows.append(dict(sample_id='b',split='test',sha='same'))
        self.assertTrue(audit.duplicate_groups(rows,'sha')[0]['cross_split'])

    def test_file_hash_and_decoded_pixels_without_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'image.png'
            image=np.zeros((12,15,3),np.uint8);image[:,:,0]=50
            self.assertTrue(cv2.imwrite(str(path),image))
            before=path.read_bytes()
            job=dict(path=str(path),kind='candidate_v3',sample_id='a',source='train',split='train',
                     expected_sha256=audit.pipeline.sha(path))
            result=audit.inspect_file(job)
            self.assertEqual(result['status'],'VERIFIED')
            self.assertEqual(result['pixel_sha_bgr'],audit.classification.pixel_sha(image))
            self.assertNotEqual(result['pixel_sha_bgr'],result['pixel_sha_rgb'])
            self.assertEqual(before,path.read_bytes())
            job['expected_sha256']='0'*64
            self.assertEqual(audit.inspect_file(job)['status'],'HASH_MISMATCH')

    def test_missing_file_and_corrupt_image_stay_unverified(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.png'
            job=dict(path=str(path),kind='candidate_v3',expected_sha256='a'*64)
            self.assertEqual(audit.inspect_file(job)['status'],'UNAVAILABLE')
            path.write_bytes(b'not an image')
            self.assertEqual(audit.inspect_file(job)['status'],'UNVERIFIED')


if __name__=='__main__':
    unittest.main()
