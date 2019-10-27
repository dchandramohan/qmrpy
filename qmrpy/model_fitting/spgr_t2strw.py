from .base_model import BaseModel
from ..signal_equations import spgr

import numpy as np
from scipy.optimize import least_squares


class SPGR_T2strw_complex(BaseModel):
    def __init__(self, TE):
        self.TE = TE # array of TE values in seconds

        self.n_fit_params = 4

        return

    def load_input_images(self, input_images):
        if not len(input_images.shape) == 4:
            raise Exception('nope!')

        if not len(self.TE) == input_images.shape[-1]:
            raise Exception('nope!')

        self.images = input_images

        return

    def residuals(self, params, obs_sig):
        M0 = params[0]
        T2str = params[1]
        df = params[2]
        phi = params[3]

        S_hat = spgr.T2strw_cplx(M0, self.TE, T2str, df, phi)

        return S_hat - obs_sig

    def fit(self, least_sq_opts=None, param_est_init=None, param_bnds=None):
        self.fit_results = np.zeros(self.images.shape[:-1] + (self.n_fit_params,))
        (nx,ny,nz) = self.images.shape[:-1] # ASSUMING 4-D images...

        calc_param_est_flag = True
        calc_bnds_flag = True

        for xi in range(nx):
            for yi in range(ny):
                for zi in range(nz):
                    if calc_param_est_flag:
                        param_est = self.calc_param_est(self.images[xi,yi,zi,:])
                    if calc_bnds_flag:
                        bnds = self.calc_bnds()

                    result = least_squares(self.residuals,
                                           param_est,
                                           bounds=bnds,
                                           args=[self.images[xi,yi,zi,:]])

                    self.fit_results[xi,yi,zi,:] = result.x

    def calc_param_est(self, voxel_signal):
        M0_est = np.max(voxel_signal)
        T2str_est = (np.log(voxel_signal[1]) - np.log(voxel_signal[0]))/(self.TE[1] - self.TE[0]) 
        
        M0_est = M0_est if (M0_est > 1.0) else 1.0
        T2str_est = T2str_est if (T2str_est > 0.0) else 0.04
        df_est = 0.0
        phi_est = 0.0

        return [M0_est, T2str_est, df_est, phi_est]

    def calc_bnds(self):
        M0_lb = 0.0
        T2str_lb = self.TE[2] - self.TE[1]
        df_lb = -1000.0
        phi_lb = -50.0

        M0_ub = 2.0
        T2str_ub = 10.0 * self.TE[-1]
        df_ub = 1000.0
        phi_ub = 50.0

        return ([M0_lb, T2str_lb, df_lb, phi_lb],
                [M0_ub, T2str_ub, df_ub, phi_ub])
