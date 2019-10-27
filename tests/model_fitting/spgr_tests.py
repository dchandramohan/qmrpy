
from ..context import signal_equations
from ..context import model_fitting

import numpy as np

class Test_SPGR_T1w_mag(unittest.TestCase):
    def test_fit(self):
        test_alpha = np.linspace(0,90,4)
        test_TR = 0.05

        test_T1 = 0.008
        test_M0 = 1.0

        sim_signal = signal_equations.spgr.T1w_mag(test_M0, test_T1, test_TR, test_alpha)
        test_image = np.reshape(sim_signal,[1,1,1,len(sim_signal)])

        fit_object = model_fitting.spgr.SPGR_T1w_mag(test_TR, test_alpha)
        fit_object.load_input_images(test_image)
        fit_object.fit()

        assert(fit_object.fit_results[1,1,1,0] == test_M0)
        assert(fit_object.fit_results[1,1,1,1] == test_T1)
