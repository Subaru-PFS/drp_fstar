import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii
from scipy.interpolate import Rbf
import parallel


class AmbreInterpolation: 
    """ Interpolate a model spectrum along four model parameters using scipy.interpolate.Rbf.
        `interpolate` method returns a `numpy.ndarray` of an interpolated spectrum.
        This script calls `parallel.py`.

    Parameters
    ----------
    teff : Effective temperature (Teff [K]) at an interpolation. type = float
    logg : Log of surface gravity (g in units of cm/s2) at an interpolation. Type = float
    metal : Mean stellar metallicity ([Metal/H] or [Fe/H]) at an interpolation. Type = float
    alpha : [alpha/Fe] chemical index at an interpolation. Type = float
    path2Spectrum : Path to a directory of spectra.
    path2List : A list of models in the directory of spectra; the list have columns of 
                [SepctrumName,Atmosphere,Teff,Logg,Mass,Turbv,Metal,Alpha]
    function : Function for RBF interpolation; the default is `multiquadric`.
    epsilon : Epsilon parameter in RBF (see scipy.interpolate.Rbf); the default is 2.0., Type = float
    """ 

    def __init__(self, teff, logg, metal, alpha, 
                 path2Spectrum, path2List, 
                 function='multiquadric', epsilon=2.0):
        self.teff = teff
        self.logg = logg
        self.metal = metal
        self.alpha = alpha
        self.path2Spectrum = path2Spectrum
        self.path2List = path2List
        self.function = function
        self.epsilon = epsilon

    def modelList(self):
        """ Read a list of spectra """
        parameterList = ascii.read(self.path2List).to_pandas()
        return parameterList
    
    def fetchReferenceModelName(self):
        """ Fetch a reference model. If no reference, return an empty numpy array. """
        parameterList = AmbreInterpolation.modelList(self)
        referenceModelName = parameterList[(parameterList['Teff'] == self.teff) & 
                                           (parameterList['Gravity'] == self.logg) & 
                                           (parameterList['Metal'] == self.metal) & 
                                           (parameterList['Alpha'] == self.alpha)]['ModelName'].values
        if referenceModelName.size != 0:
            return referenceModelName[0]
        if referenceModelName.size == 0:
            return np.array([])

    def interpolate(self, nProcs=1):        
        """ Do interpolation

        Return an interpolated spectrum and names of models used for the interpolation.

        Usage
        ---------
        interpolation, neighborModels = interpolate(nProcs)

        Parameter
        ---------
        --nProcs : Number of processes for multiprocessing; the default is 1 (sigle process). Type = int
        """
        def fetchNeighborModels(self, indexTeffRange, indexLoggRange, indexMetalRange, indexAlphaRange, 
                                referenceModelName):
            """ Fetch a dataset of model names and parameters of neighbor models.
            If there is a reference model, it is removed from the neighbor model list.
            """
            parameterList = AmbreInterpolation.modelList(self)
            if len(referenceModelName) > 0: 
                neighborModel = parameterList[(parameterList['Teff'] >= self.teff - teffRange[indexTeffRange])
                                     & (parameterList['Teff'] <= self.teff + teffRange[indexTeffRange])
                                     & (parameterList['Gravity'] >= self.logg - loggRange[indexLoggRange])  
                                     & (parameterList['Gravity'] <= self.logg + loggRange[indexLoggRange])  
                                     & (parameterList['Metal'] >= self.metal - metalRange[indexMetalRange]) 
                                     & (parameterList['Metal'] <= self.metal + metalRange[indexMetalRange]) 
                                     & (parameterList['Alpha'] >= self.alpha - alphaRange[indexAlphaRange]) 
                                     & (parameterList['Alpha'] <= self.alpha + alphaRange[indexAlphaRange]) 
                                     & (parameterList['ModelName'] != referenceModelName)]
                return neighborModel
            if len(referenceModelName) == 0:
                neighborModel = parameterList[(parameterList['Teff'] >= self.teff - teffRange[indexTeffRange])
                                     & (parameterList['Teff'] <= self.teff + teffRange[indexTeffRange])
                                     & (parameterList['Gravity'] >= self.logg - loggRange[indexLoggRange])  
                                     & (parameterList['Gravity'] <= self.logg + loggRange[indexLoggRange])  
                                     & (parameterList['Metal'] >= self.metal - metalRange[indexMetalRange]) 
                                     & (parameterList['Metal'] <= self.metal + metalRange[indexMetalRange]) 
                                     & (parameterList['Alpha'] >= self.alpha - alphaRange[indexAlphaRange]) 
                                     & (parameterList['Alpha'] <= self.alpha + alphaRange[indexAlphaRange])]
                return neighborModel

        referenceModelName = AmbreInterpolation.fetchReferenceModelName(self)

        ## step of Teff parameters
        teffRange = [500, 750, 1000, 1250, 1500, 1750, 2000]
        loggRange = [1.0, 1.5, 2.0, 2.5]
        metalRange = np.arange(0.5, 4.75, 0.25)
        alphaRange = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        ## initial set
        numOfModelsAlongTeff = numOfModelsAlongLogg = numOfModelsAlongMetal = numOfModelsAlongAlpha = 0
        indexTeffRange = indexLoggRange = indexMetalRange = indexAlphaRange = 0
        requiredNumberOfParameters = 3

        ## Search ranges of parameters which enclos a sepcified number of steps ('requiredNumberOfParameters')
        while numOfModelsAlongTeff < requiredNumberOfParameters + 1:
            neighborModel = fetchNeighborModels(self, indexTeffRange, indexLoggRange, 
                                                indexMetalRange, indexAlphaRange, referenceModelName)
            numOfModelsAlongTeff = len(neighborModel['Teff'].unique())
            if numOfModelsAlongTeff >= requiredNumberOfParameters + 1: 
                break
            indexTeffRange = indexTeffRange + 1
            if indexTeffRange > len(teffRange) -1:
                indexTeffRange = indexTeffRange -1
                break
        while numOfModelsAlongLogg < requiredNumberOfParameters:
            neighborModel = fetchNeighborModels(self, indexTeffRange, indexLoggRange, indexMetalRange, 
                                                indexAlphaRange, referenceModelName)
            numOfModelsAlongLogg = len(neighborModel['Gravity'].unique())
            if numOfModelsAlongLogg >= requiredNumberOfParameters: 
                break
            indexLoggRange = indexLoggRange + 1
            if indexLoggRange > len(loggRange) -1:
                indexLoggRange = indexLoggRange -1
                break
        while numOfModelsAlongMetal < requiredNumberOfParameters:
            neighborModel = fetchNeighborModels(self, indexTeffRange, indexLoggRange, indexMetalRange, 
                                                indexAlphaRange, referenceModelName)
            numOfModelsAlongMetal = len(neighborModel['Metal'].unique())
            if numOfModelsAlongMetal >= requiredNumberOfParameters: 
                break
            indexMetalRange = indexMetalRange + 1
            if indexMetalRange > len(metalRange) -1:
                indexMetalRange = indexMetalRange -1
                break
        while numOfModelsAlongAlpha < requiredNumberOfParameters:
            neighborModel = fetchNeighborModels(self, indexTeffRange, indexLoggRange, indexMetalRange, 
                                                indexAlphaRange, referenceModelName)
            numOfModelsAlongAlpha = len(neighborModel['Alpha'].unique())
            if numOfModelsAlongAlpha >= requiredNumberOfParameters: 
                break
            indexAlphaRange = indexAlphaRange + 1
            if indexAlphaRange > len(alphaRange) -1:
                indexAlphaRange = indexAlphaRange -1
                break

        if ((numOfModelsAlongTeff >= requiredNumberOfParameters) & 
            (numOfModelsAlongLogg >= requiredNumberOfParameters) & 
            (numOfModelsAlongMetal >= requiredNumberOfParameters) &
            (numOfModelsAlongAlpha >= requiredNumberOfParameters)): 
            neighborModel = fetchNeighborModels(self, indexTeffRange, indexLoggRange, indexMetalRange, 
                                                indexAlphaRange, referenceModelName)
            
            neighborModelParameter = np.array(neighborModel[['Teff','Gravity', 'Metal', 'Alpha']])
            neighborModelName = neighborModel['ModelName']
            
            with fits.open(self.path2Spectrum + neighborModelName.reset_index(drop=True)[0]) as tempModelFITS:
                tempModel = Table(tempModelFITS[1].data).to_pandas()
            neighborModelFlux = np.empty((len(tempModel), len(neighborModelName)), dtype='float64')
#            neighborModelFlux = np.empty((1000, len(neighborModelName)), dtype='float64')
            
            for i, row in enumerate(neighborModelName):
                with fits.open(self.path2Spectrum + row) as modelFITS:
                    model = Table(modelFITS[1].data).to_pandas()
                neighborModelFlux[:,i] = model['Flux']
#                neighborModelFlux[:,i] = model['Flux'][:1000]
            
            def rbfInterpolate(neighborModelFlux):
                interpolationFunction = Rbf(neighborModelParameter[:,0]/1e3, neighborModelParameter[:,1], 
                                            neighborModelParameter[:,2], neighborModelParameter[:,3], 
                                            neighborModelFlux, function=self.function, epsilon=self.epsilon)
                interpolatedFlux = interpolationFunction(self.teff/1e3, self.logg, self.metal, self.alpha)
                return interpolatedFlux

            result = parallel.parallel_map(rbfInterpolate, neighborModelFlux, n_procs=nProcs)
            result = np.hstack(result)           
            return result, neighborModelName.values

        else:
            print('Not enough number of parameters. This process ends')  
            return np.array([])


