import numpy as np
import pandas as pd  
from astropy.io import fits
from astropy.table import Table     
from astropy.io import ascii       
from scipy.interpolate import Rbf 
import parallel
import pickle


""" Set input information """
## inputDir: Directory name of input spectra.
inputDir = '/data22b/work/takuji/AMBRE/extrapolation20190419/extrapolated_all/'
## inputList: Ascii table listing parameters to be used for interpolation.
inputList = '/home/takuji/work/ambre/jn/extrapolatedmodel_parameters_p5500-8000.dat'
## output: Output file name. 
output = 'ambreRBF.pickle'
## n_procs: int. Number of processes. 
n_procs = 7


def makeRBFInterpolationModel(inputDir, inputList, output, n_procs):
    inputModelList = ascii.read(inputList).to_pandas()
    
    ## fetch columns used for making a model
    inputParameter = np.array(inputModelList[['Teff', 'Logg', 'Z', 'Alpha']])
   
    ## make an empty array for input fluxes
    with fits.open(inputDir + inputModelList['ModelName'].reset_index(drop=True)[0]) as dummyFITS:
        dummyModel = Table(dummyFITS[1].data).to_pandas()
        dummyheader = dummyFITS[1].header
    inputFlux = np.empty((len(dummyModel), len(inputModelList['ModelName'].values)), dtype='float64')

    ## Store fluxes of each model into a single array
    for i, row in enumerate(inputModelList['ModelName'].values):
        with fits.open(inputDir + row) as modelFITS:
            model = Table(modelFITS[1].data).to_pandas()
            inputFlux[:,i] = model['Flux']

    ## Function for making an RBF model
    def makeRBFModel(inputflux):
        interpolationFunction = Rbf(inputParameter[:,0]/1e3, ## Scale down Teff by 1/1000
                                    inputParameter[:,1],
                                    inputParameter[:,2],
                                    inputParameter[:,3],
                                    inputflux,
                                    function='multiquadric', 
                                    epsilon=2.0)
        return interpolationFunction
    
    inputFlux = inputFlux[:10,:]
   
    ## Execute `makeRBFModel` in parallel processsing
    interpolationFunction = parallel.parallel_map(makeRBFModel, inputFlux, n_procs)

    ## Save the pickle model
    with open(output, 'wb') as savePickle:
        pickle.dump(interpolationFunction, savePickle)

if __name__ == '__main__':
    makeRBFInterpolationModel(inputDir, inputList, output, n_procs)

