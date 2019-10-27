
import numpy as np
from scipy.optimize import least_squares

from .base_model import BaseModel
from ..signal_equations import spgr

class SPGR_T1w_mag(BaseModel):

    def __init__(self, TR, alpha):
        self.TR = TR # TR is a single value in seconds
        self.alpha = alpha # Flip angle is an array of flip angles in degrees corresponding to the number of input images

        self.n_fit_params = 2

        return

    def load_input_images(self, input_images):
        if not input_images.shape[-1] == len(self.alpha):
            raise Exception('Hey there! this is no good!')

        self.images = input_images

    def residuals(self, params, obs_sig):
        K = params[0]
        T1 = params[1]

        S_hat = spgr.T1w_mag(K, T1, self.TR, self.alpha)
        return S_hat - obs_sig

    def fit(self, least_sq_opts=None, param_est_init=None, param_bnds=None):
        self.fit_results = np.zeros(list(self.images.shape[:-1])+[self.n_fit_params])
        (nx,ny,nz) = self.images.shape[:-1]

        # implement supplying estimates and bounds later
        calc_param_est_flag = True
        calc_bnds_flag = True

        for xi in range(nx):
            for yi in range(ny):
                for zi in range(nz):
                    if len(np.array(param_est_init).shape) == 1:
                        assert len(param_est_init) == self.n_fit_params, 'Incorrect number of parameter estimates'
                        param_est = param_est_init
                    else if param_est_init.shape == (nx, ny, nz):
                        param_est = param_est_init[xi, yi, zi]
                    else if calc_param_est_flag:
                        # M0_est = np.max(np.squeeze(self.images[xi,yi,zi,:]))
                        # S1 = self.images[xi,yi,zi,0]
                        # S2 = self.images[xi,yi,zi,1]
                        #
                        # fa1 = self.alpha[0]
                        # fa2 = self.alpha[1]
                        #
                        # T1_est = -1.0 * self.TR / np.log((S1/np.sin(fa1) - S2/np.sin(fa2)) / (S1/np.tan(fa1) - S2/np.tan(fa2)))
                        #
                        # param_est = [M0_est, T1_est]
                        param_est = self.calc_param_est()

                    if len(np.array(param_bnds[0]).shape) == 1:
                        assert len(np.array(param_bnds[0])) == len(np.array(param_bnds[1])) == self.n_fit_params, \
                            'Incorrect number of parameter estimates'
                        bnds = param_bnds
                    else if param_bnds[0].shape == param_bnds[1].shape == (nx, ny, nz):
                        bnds = param_bnds[xi, yi, zi]
                    else if calc_bnds_flag:
                        bnds = self.calc_bounds()
                    else:
                        assert param_bnds == None, 'Invalid parameter bounds'

                    result = least_squares(self.residuals,
                                           param_est,args=[self.images[xi,yi,zi,:]],bounds=bnds)

                    self.fit_results[xi,yi,zi,:] = result.x

        return

    def calc_param_est(self):
        # fill in
        pass

    def calc_bounds(self):
        # fill in
        pass
