import json
import unittest

import numpy as np
import pandas as pd

import interference_metrics as m


def predictions(n=8, model='B', seed=42):
    return pd.DataFrame([dict(seed=seed, model=model, sample_id=f'v{k}', video=f'{k}.mp4',
        video_sha256=f'video{k}', image_sha256=f'image{k}', true_class_id=k,
        true_class=m.ALL_CLASSES[k], probabilities=json.dumps(np.eye(n)[k].tolist())) for k in range(n)])


class InterferenceTests(unittest.TestCase):
    def test_mapping_alignment_and_missing_predictions(self):
        a, b = predictions(6, 'A'), predictions()
        y, pa, pb = m.align_predictions(a.iloc[::-1], b, range(6))
        np.testing.assert_array_equal(y, np.arange(6))
        np.testing.assert_array_equal(pa.argmax(1), pb.argmax(1))
        for bad in [pd.concat([b, b.iloc[:1]]), b.iloc[1:]]:
            with self.assertRaises(ValueError): m.align_predictions(a, bad, range(6))
        for field, value in [('true_class', 'wrong'), ('video', '0.mp4'), ('video_sha256', 'video0')]:
            bad = b.copy(); bad.loc[1, field] = value
            with self.assertRaises(ValueError): m.align_predictions(a, bad, range(6))

    def test_full_argmax_intrusion_recall_and_identity(self):
        y = np.arange(6); a = np.eye(6); b = np.pad(a, ((0, 0), (0, 2)))
        b[0] = [0.3, 0, 0, 0, 0, 0, 0.7, 0]
        b[1] = [0.3, 0.2, 0, 0, 0, 0, 0, 0.5]
        out = m.interference(y, a, b)
        self.assertAlmostEqual(out['intrusion'][0]['overall'], 2/6)
        self.assertAlmostEqual(out['intrusion'][0]['halo'], 1/6)
        self.assertAlmostEqual(out['intrusion'][0]['tidur'], 1/6)
        self.assertEqual(out['classes'][0]['delta_recall'], -1)
        d = out['decomposition']
        self.assertAlmostEqual(d['gap_A_minus_B'], d['restricted_gap'] + d['intrusion_gap'])
        self.assertAlmostEqual(d['accuracy_B_restricted'], 5/6)
        self.assertAlmostEqual(d['accuracy_B'], 4/6)

    def test_missing_class_and_invalid_scores_rejected(self):
        with self.assertRaises(ValueError): m.metrics([0, 1], [0, 1], range(6))
        for bad in [np.full((6, 8), np.nan), np.zeros((6, 8)), np.ones((6, 7))/7]:
            with self.assertRaises(ValueError): m.interference(np.arange(6), np.eye(6), bad)
        with self.assertRaises(ValueError): m.metrics([0.5], [0], range(1))

    def test_full8_precision_f1_confusion_counts(self):
        r = m.metrics(np.arange(8), np.arange(8), range(8))
        self.assertEqual(r['macro_f1'], 1)
        self.assertEqual(r['accuracy'], 1)
        self.assertEqual(r['confusion_matrix'], np.eye(8, dtype=int).tolist())
        self.assertTrue(all(x['n'] == 1 and x['precision'] == x['recall'] == x['f1'] == 1 for x in r['classes']))


if __name__ == '__main__': unittest.main()
