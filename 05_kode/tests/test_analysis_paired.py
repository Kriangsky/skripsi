import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

import analysis_paired as a


def fixture(work):
    manifest=[];predictions=[]
    for cid in range(6):
        for i in range(2):
            sid=f'{cid}-{i}'
            image=np.full((4,4,3),cid*20+i,dtype=np.uint8)
            for model,n in [('A',6),('B',8)]:
                folder=work/'datasets'/model/'val'/a.c.folder_names(n)[cid]
                folder.mkdir(parents=True,exist_ok=True)
                path=folder/f'{sid}.png';cv2.imwrite(str(path),image)
                predicted=6 if model=='B' and cid==0 and i==0 else cid
                scores=np.eye(n)[predicted].tolist()
                predictions.append(dict(model=model,sample_id=sid,true_class_id=cid,true_class=a.c.CLASSES[cid],
                    predicted_class_id=predicted,predicted_class=a.c.CLASSES[predicted],confidence=1.,
                    probabilities=json.dumps(scores),quality_flags='[]'))
            manifest.append(dict(sample_id=sid,video=f'train/{cid}/{sid}.mp4',video_sha256=sid,split='val',
                source='train',class_id=cid,class_name=a.c.CLASSES[cid],image_sha256=a.c.b.sha(path)))
    (work/'evaluation').mkdir()
    (work/'runs').mkdir()
    (work/'runs/trained_models.json').write_text('{}')
    pd.DataFrame(manifest).to_csv(work/'manifest.csv',index=False)
    pd.DataFrame(predictions).to_csv(work/'evaluation/predictions.csv',index=False)


class PairedAnalysisTests(unittest.TestCase):
    def test_markdown_without_optional_dependencies(self):
        text=a.markdown_table(pd.DataFrame({'label':['a|b'],'value':[1]}))
        self.assertIn('a\\|b',text)
        self.assertIn('| --- | --- |',text)

    def test_exact_discordants_zero_and_known_case(self):
        self.assertEqual(a.mcnemar_exact(0,0),1.)
        self.assertEqual(a.mcnemar_exact(1,0),1.)
        self.assertEqual(a.mcnemar_exact(6,0),.03125)
        with self.assertRaises(ValueError):a.mcnemar_exact(-1,0)

    def test_wilson_boundaries(self):
        lo,hi=a.wilson(0,50);self.assertEqual(lo,0);self.assertGreater(hi,0)
        lo,hi=a.wilson(50,50);self.assertEqual(hi,1);self.assertLess(lo,1)
        self.assertAlmostEqual(a.wilson(25,50)[0]+a.wilson(25,50)[1],1)
        with self.assertRaises(ValueError):a.wilson(1,0)

    def test_bootstrap_pairing_reproducibility_and_extra_outputs(self):
        y=np.repeat(np.arange(6),3);b=y.copy();b[0]=6
        x=a.paired_bootstrap(y,y,b,100)
        np.testing.assert_array_equal(x,a.paired_bootstrap(y,y,b,100))
        self.assertTrue((x[:,1]<=0).all())
        np.testing.assert_array_equal(a.paired_bootstrap(y,y,y,100),np.zeros((100,2)))
        with self.assertRaises(ValueError):a.paired_bootstrap([0],[0],[0])

    def test_join_hash_gt_integrity_and_reports_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'run';work.mkdir();fixture(work)
            pairs=a.load_pairs(work);self.assertEqual(len(pairs),12)
            record=a.analyze(work,root/'analysis',100)
            self.assertEqual(record['counts']['a_only'],1)
            self.assertEqual(len(pd.read_csv(root/'analysis/paired_errors.csv')),1)
            self.assertIsNone(record['seed_variance'])
            with self.assertRaises(FileExistsError):a.analyze(work,root/'analysis',100)
            data=pd.read_csv(work/'evaluation/predictions.csv')
            data.loc[0,'true_class_id']=1;data.to_csv(work/'evaluation/predictions.csv',index=False)
            with self.assertRaises(ValueError):a.load_pairs(work)

    def test_corrupt_shared_image_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);fixture(work)
            next((work/'datasets/A').rglob('*.png')).write_bytes(b'changed')
            with self.assertRaises(ValueError):a.load_pairs(work)


if __name__=='__main__':unittest.main()
