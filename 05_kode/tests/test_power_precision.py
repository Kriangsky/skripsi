from pathlib import Path
import tempfile
import unittest

import numpy as np

import power_precision as p
from test_analysis_paired import fixture


class PowerPrecisionTests(unittest.TestCase):
    def test_joint_mass_calibration_and_extra_outputs(self):
        joint=p.calibrated_joint(.05,0.)
        np.testing.assert_allclose(joint.sum(axis=(1,2)),1)
        self.assertGreater(joint[:,:,6:].sum(),0)
        self.assertAlmostEqual(p.expected_delta(joint),0,places=9)
        with self.assertRaises(ValueError):p.calibrated_joint(.001,-.1)

    def test_gradient_matches_finite_difference(self):
        joint=p.calibrated_joint(.05,-.01)
        matrix=joint.sum(axis=1)/6;g=p.f1_gradient(matrix)
        for index in [(0,0),(2,6),(3,2)]:
            changed=matrix.copy();changed[index]+=1e-7
            numeric=(p.f1_from_matrix(changed)-p.f1_from_matrix(matrix))/1e-7
            self.assertAlmostEqual(g[index],numeric,places=5)

    def test_variance_sample_size_and_seed_sensitivity(self):
        joint=p.calibrated_joint(.05,0)
        self.assertAlmostEqual(p.video_variance(joint,50)/p.video_variance(joint,100),2)
        a=p.simulate_h1(joint,0,300);b=p.simulate_h1(joint,.02,300)
        self.assertGreater(b['ci95_width_pp'],a['ci95_width_pp'])
        self.assertEqual(a,p.simulate_h1(joint,0,300))
        self.assertTrue(0<=a['ni_pass_probability']<=1)

    def test_h2_exact_binomial_and_joint_assumption(self):
        a=p.h2_binomial(.8);b=p.h2_binomial(.95)
        self.assertLessEqual(a['probability_pass_one_class'],.05)
        self.assertGreater(b['probability_pass_one_class'],a['probability_pass_one_class'])
        self.assertAlmostEqual(b['probability_pass_both_if_independent'],b['probability_pass_one_class']**2)
        self.assertGreaterEqual(b['required_correct'],41)
        self.assertEqual(p.h2_binomial(1)['probability_pass_one_class'],1.)

    def test_report_labels_simulation_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'run';work.mkdir();fixture(work)
            h1,h2=p.run(work,root/'power',50)
            self.assertIn('HYPOTHETICAL',set(h1.scenario))
            self.assertEqual(len(h2),5)
            text=(root/'power/power_precision_report.md').read_text()
            self.assertIn('not estimated external behavior',text)
            with self.assertRaises(FileExistsError):p.run(work,root/'power',50)


if __name__=='__main__':unittest.main()
