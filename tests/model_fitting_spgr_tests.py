import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import qmrpy.model_fitting.spgr
import qmrpy.signal_equations.spgr

import unittest
import numpy as np
import numpy.testing as npt

class Test_SPGR_T1w_mag(unittest.TestCase):
    def test_fit(self):
        test_alpha = np.linspace(6,90,10)
        test_TR = 0.05

        test_M0 = 1.0
        test_T1 = 0.008

        sim_signal = qmrpy.signal_equations.spgr.T1w_mag(test_M0, test_T1, test_TR, test_alpha)
        test_image = np.reshape(sim_signal,[1,1,1,len(sim_signal)])

        fit_object = qmrpy.model_fitting.spgr.SPGR_T1w_mag(test_TR, test_alpha)
        fit_object.load_input_images(test_image)
        fit_object.fit()

        npt.assert_allclose(fit_object.fit_results[0,0,0,:],np.array([test_M0,test_T1]))

    def test_fit_est_and_bnds(self):
        test_alpha = np.linspace(6,90,10)
        test_TR = 0.05

        test_M0 = 1.0
        test_T1 = 0.008

        test_param_est = [1, 0.8]
        test_param_bounds = ([0,0], [2,5])

        sim_signal = qmrpy.signal_equations.spgr.T1w_mag(test_M0, test_T1, test_TR, test_alpha)
        test_image = np.reshape(sim_signal,[1,1,1,len(sim_signal)])

        fit_object = qmrpy.model_fitting.spgr.SPGR_T1w_mag(test_TR, test_alpha)
        fit_object.load_input_images(test_image)
        fit_object.fit(param_est_init = test_param_est, param_bnds = test_param_bounds)

        npt.assert_allclose(fit_object.fit_results[0,0,0,:],np.array([test_M0,test_T1]))
