import numpy as np
import parallel
import pickle

class InterpolationModel:
    def __init__(self, modelPickle):
        self.modelPickle = modelPickle

    def readRBFModel(self):
        with open(self.modelPickle, 'rb') as readPickle:
            interpolationModel = pickle.load(readPickle)
        return interpolationModel
    

class Interpolation:
    def __init__(self):
        pass

    def interpolate(self, model, teff, logg, metal, alpha, nProcs):
        def doInterpolation(model):
            interpolationFunction = model 
            interpolatedFlux = interpolationFunction(teff/1e3, logg, metal, alpha)
            return interpolatedFlux
        spectrum = parallel.parallel_map(doInterpolation, model, n_procs=nProcs)
        return np.hstack(spectrum)

