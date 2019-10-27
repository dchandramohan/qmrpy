
import abc

class BaseModel(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def residuals(self, params, obs_sig):
        pass
    
    @abc.abstractmethod
    def fit(self, least_sq_opts=None, **kwargs):
        pass

    @abc.abstractmethod
    def load_input_images(self, input_images):
        pass

    def plot_fit_results(self):
        if self.fit_results == None:
            raise Exception('fitting must be done first')

        # implement here
        for fit_var in fit_results:
            # do something to plot fit_results[fit_var] which should be a 3d-array of input image size
            pass

    def plot_single_voxel_curve(self):
        # MAYBE WE DO THIS
        pass
