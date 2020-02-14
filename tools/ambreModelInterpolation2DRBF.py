import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii
from scipy.interpolate import Rbf
import parallel


class Ambre2drbfInterpolation:
    def __init__(self, teff, logg, metal, alpha,
        path2Spectrum, originalParameters,
        function='multiquadric', epsilon=2.0):
        self.teff = teff
        self.logg = logg
        self.metal = metal
        self.alpha = alpha
        self.path2Spectrum = path2Spectrum
        self.originalParameters = originalParameters
        self.function = function
        self.epsilon = epsilon
        
    def originalParam(self):
        """ Read a list of original spectra """
        originalParameterList = ascii.read(self.originalParameters).to_pandas()
        originalGrid = originalParameterList[(originalParameterList['Z'] == self.metal) & 
                                             (originalParameterList['Alpha'] == self.alpha)]
        duplication = originalGrid[(originalGrid['Teff'] == self.teff) &
                                   (originalGrid['Logg'] == self.logg) &
                                   (originalGrid['Z'] == self.metal) &
                                   (originalGrid['Alpha'] == self.alpha)]
        if (len(duplication) > 0):
            originalGrid = originalGrid.drop(duplication.index)
        return originalGrid

    def originalParamNum(self):
        """ Count a number of original grid points """
        return len(Ambre2drbfInterpolation.originalParam(self))

    def interpolate(self, nProcs=1):
        """ Do interpolation
        Return an interpolated spectrum
        Usage: interpolation = interpolate(nProcs)
        Parameter: 
        --nProcs : Number of processes for multiprocessing; the default is 1 (sigle process). Type = int
        """

        originalGrid = Ambre2drbfInterpolation.originalParam(self)

        if (len(originalGrid) < 1):
            print('Waring: No original models for interpolation.')
        else:
            ## make an empty array and store original models
            with fits.open(self.path2Spectrum + originalGrid['ModelName'].reset_index(drop=True)[0]
                          ) as tempModelFITS:
                tempModel = Table(tempModelFITS[1].data).to_pandas()
            neighborModelFlux = np.empty((len(tempModel), len(originalGrid)), dtype='float64')
            
            for i, template in enumerate(originalGrid['ModelName']):
                with fits.open(self.path2Spectrum + template) as modelFITS:
                    model = Table(modelFITS[1].data).to_pandas()
                neighborModelFlux[:,i] = model['Flux']
            
            ## interpolation
            def rbf2dInterpolate(neighborModelFlux):
                interpolationFunction = Rbf(originalGrid['Teff']/1e3, 
                                                originalGrid['Logg'], 
                                                neighborModelFlux, 
                                                function=self.function, epsilon=self.epsilon)
                interpolatedFlux = interpolationFunction(self.teff/1e3, self.logg)
                return interpolatedFlux
                
            result = parallel.parallel_map(rbf2dInterpolate, neighborModelFlux, n_procs=nProcs)
            interpolatedSpectrum = np.hstack(result)        
            return interpolatedSpectrum

