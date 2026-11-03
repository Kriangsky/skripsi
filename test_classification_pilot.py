from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

import classification_pilot as c


def inventory():
    return pd.DataFrame([dict(sample_id=f'{n}-{split}-{i}', video=f'train/{name}/{split}{i}.mp4',
        source='train', split=split, class_name=name, class_id=n, signer='',
        video_sha256=f'{n}-{split}-{i}') for n,name in enumerate(c.CLASSES)
        for split,count in [('train',65),('val',25)] for i in range(count)])


class ClassificationPilotTests(unittest.TestCase):
    def test_sampling_reproducible_balanced_metadata_not_a_gate(self):
        data=inventory()
        a=c.sample_videos(data,set())
        b=c.sample_videos(data.sample(frac=1,random_state=7),set())
        self.assertEqual(a.sample_id.tolist(),b.sample_id.tolist())
        self.assertEqual(len(a),640)
        self.assertTrue(a.groupby(['class_name','split']).size().unstack().train.eq(60).all())
        self.assertTrue(a.groupby(['class_name','split']).size().unstack().val.eq(20).all())
        self.assertEqual(len(a[a.class_name.isin(c.BASE)]),480)

    def test_external_and_development_validation_excluded(self):
        data=inventory()
        external=data.iloc[:1].assign(source='test',split='test',sample_id='external',video='test/external.mp4',video_sha256='external')
        sampled=c.sample_videos(pd.concat([data,external]),set())
        self.assertNotIn('external',set(sampled.sample_id))
        development=set(data[data.split.eq('val')].sample_id.iloc[:2])
        sampled=c.sample_videos(data,development)
        self.assertFalse(set(sampled[sampled.split.eq('val')].sample_id)&development)
        with self.assertRaises(ValueError):c.sample_videos(data,set(data[data.split.eq('val')].sample_id))

    def test_class_mapping_and_duplicate_identity_rejected(self):
        data=inventory();data.loc[0,'class_id']=7
        with self.assertRaises(ValueError):c.sample_videos(data,set())
        data=inventory();data.loc[1,'video_sha256']=data.loc[0,'video_sha256']
        with self.assertRaises(ValueError):c.sample_videos(data,set())
        self.assertEqual(c.folder_names(6),['00_saya','01_kamu','02_senang','03_bingung','04_marah','05_apa_kabar'])
        self.assertEqual(c.folder_names(8)[:6],c.folder_names(6))

    def test_letterbox_preserves_edges_and_aspect_no_crop(self):
        image=np.zeros((100,200,3),dtype=np.uint8)
        image[:,:4]=[0,0,255];image[:,-4:]=[0,255,0]
        square=c.letterbox(image)
        self.assertEqual(square.shape,(224,224,3))
        self.assertTrue(np.any(square[:,:,2]==255))
        self.assertTrue(np.any(square[:,:,1]==255))
        self.assertTrue(np.all(square[0]==114))

    def test_metrics_keep_model_b_extra_predictions_common6(self):
        metrics,per_class,matrix=c.metrics([0,0,1,1],[0,6,1,7],8,6)
        self.assertEqual(metrics['accuracy'],.5)
        self.assertEqual(metrics['extra_class_predictions_n'],2)
        self.assertEqual(matrix[0][6],1)
        self.assertEqual(matrix[1][7],1)
        self.assertAlmostEqual(metrics['macro_f1'],(2/3+2/3)/6)
        self.assertEqual(sum(map(sum,matrix)),4)

    def test_audit_balanced_additional_flagged_and_no_exclusion(self):
        data=c.sample_videos(inventory(),set()).assign(quality_status='flagged',quality_flags='["possible_blur"]')
        audit=c.audit_sample(data)
        self.assertTrue(audit.groupby('class_name').size().eq(20).all())
        self.assertEqual(len(audit),160)
        self.assertEqual(len(data),640)
        self.assertEqual(audit.sample_id.tolist(),c.audit_sample(data.sample(frac=1)).sample_id.tolist())

    def test_corrupt_release_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            (work/'reports').mkdir()
            image=work/'sample.png';image.write_bytes(b'broken')
            c.dump(work/'release.json',dict(files={'sample.png':c.b.sha(image)},fingerprint='fixture'))
            image.write_bytes(b'changed')
            with self.assertRaises(ValueError):c.validate_release(work)

    def test_final_validation_callback_does_not_add_training_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            pd.DataFrame(dict(epoch=range(1,31))).to_csv(folder/'results.csv',index=False)
            self.assertEqual(c.completed_epochs(SimpleNamespace(save_dir=folder,epoch=30)),30)


if __name__=='__main__':unittest.main()
