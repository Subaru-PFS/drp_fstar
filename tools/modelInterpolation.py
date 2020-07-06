import numpy as np
import parallel
import pickle

class InterpolationModel:
    """ Read an RBF picke file model 
    
    Parameters
    ----------
    modelPickle : 'str'
       File name of an RBF model generated through `makeRBFInterpolationModel.py`.

    Returns
    -------
    interpolationModel : pickle module
       The RBF model is read as pickle module.
    """

    def __init__(self, modelPickle):
        self.modelPickle = modelPickle ## a path to the RBF model

    def readRBFModel(self):
        """ Return a list of Rbf objects """
        with open(self.modelPickle, 'rb') as readPickle:
            interpolationModel = pickle.load(readPickle)
        return interpolationModel
    

class Interpolation:
    """ Generate an interpolated spectrum at a given parameter point 

    Parameters
    ----------
    model : pickle module
       RBF model that was read with `InterpolationModel()`.
    teff : `float`
       Effective temepature in K for interpolation.
    logg : `float`
       Surface gravity in log(/(cm/s^2)) for interpolation.
    metal : `float`
       Metallicity [Fe/H] for interpolation.
    alpha : `float`
       Alpha element index [alpha/Fe] for interpolation.
    nProcs : 'int`
       A number of processes.
 
    Returns
    -------
    spectrum : `numpy.ndarray`
       Interpolation spectrum.
    """

    def __init__(self):
        pass

    def interpolate(self, model, teff, logg, metal, alpha, nProcs):
        def doInterpolation(model):
            interpolationFunction = model 
            interpolatedFlux = interpolationFunction(teff/1e3, logg, metal, alpha)
            return interpolatedFlux
        spectrum = parallel.parallel_map(doInterpolation, model, n_procs=nProcs)
        return np.hstack(spectrum)
