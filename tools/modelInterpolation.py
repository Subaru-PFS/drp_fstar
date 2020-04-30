import numpy as np
import parallel
import pickle

class InterpolationModel:
    """ Read an RBF picke file model """
    def __init__(self, modelPickle):
        self.modelPickle = modelPickle ## a path to the RBF model

    def readRBFModel(self):
        """ Return a list of Rbf objects """
        with open(self.modelPickle, 'rb') as readPickle:
            interpolationModel = pickle.load(readPickle)
        return interpolationModel
    

class Interpolation:
    """ Generate an interpolated spectrum at a given parameter point """
    def __init__(self):
        pass

    def interpolate(self, model, teff, logg, metal, alpha, nProcs):
        """ 
        Parameters
        ----------
        model : Rbf object list 
        teff : effective temepature in Kelvin for interpolation
        logg : surface gravity in log(/(cm/s^2)) for interpolation
        metal : metallicity [Fe/H] for interpolation
        alpha : alpha element index [alpha/Fe] for interpolation
        nProcs : a number of processes 
        """
        def doInterpolation(model):
            interpolationFunction = model 
            interpolatedFlux = interpolationFunction(teff/1e3, logg, metal, alpha)
            return interpolatedFlux
        spectrum = parallel.parallel_map(doInterpolation, model, n_procs=nProcs)
        return np.hstack(spectrum)
