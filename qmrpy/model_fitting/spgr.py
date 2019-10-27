
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

        # put parameter estimate code here
        param_est_M0 = None;
        param_est_T1 = None;

        param_est = [param_est_M0, param_est_T1]
        
        # put optional upper and lower bound code here
        lower_bounds = [None, None]
        upper_bounds = [None, None] # ?????? do this?

        for xi in range(nx):
            for yi in range(ny):
                for zi in range(nz):
                    result = least_squares(self.residuals,
                                           param_est)

                    self.fit_results[xi,yi,zi,:] = result.x

        return
    
        
