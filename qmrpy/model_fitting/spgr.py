
from .base_model import BaseModel
from ..signal_equations import spgr
from scipy.optimize import least_squares

class SPGR_T1w_mag(BaseModel):

    def __init__(self, TR, alpha):
        self.TR = TR # TR is a single value in seconds
        self.alpha = alpha # Flip angle is an array of flip angles in degrees corresponding to the number of input images

        self.n_fit_params = 2

        return

    def load_input_images(self, input_images):
        if not input_images.shape[-1] == len(alpha):
            raise Exception('Hey there! this is no good!')

        self.images = input_images

    def residuals(self, params, obs_sig):
        K = params[0]
        T1 = params[1]
        
        S_hat = spgr.T1w_mag(K, T1, self.TR, self.alpha)
        return S_hat - obs_sig

    def fit(self, **kwargs, least_sq_opts=None):
        self.fit_results = np.zeros(self.images.shape[:-1]+[self.n_fit_params])
        (nx,ny,nz) = self.images.shape[:-1]

        # implement supplying estimates and bounds later
        calc_param_est_flag = True
        calc_bnds_flag = True
        
        for xi in range(nx):
            for yi in range(ny):
                for zi in range(nz):
                    if calc_param_est_flag:
                        M0_est = np.max(np.squeeze(self.images[xi,yi,zi,:]))
                        S1 = self.images[xi,yi,zi,0]
                        S2 = self.images[xi,yi,zi,1]

                        fa1 = self.alpha[0]
                        fa2 = self.alpha[1]
                        
                        T1_est = -1.0 * TR / np.log((S1/np.sin(self.fa1) - S2/np.sin(fa2)) / (S1/np.tan(fa1) - S2/np.tan(fa2)))

                        param_est = [M0_est, T1_est]
                        
                    result = least_squares(self.residuals,
                                           param_est)

                    self.fit_results[xi,yi,zi,:] = result.x

        return
    
        
