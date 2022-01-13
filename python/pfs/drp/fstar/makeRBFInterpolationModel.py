import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii
from scipy.interpolate import Rbf

import pickle
import argparse

from . import parallel


def makeRBFInterpolationModel(inputDir, inputList, outputModel, n_procs):
    """Generate an RBF interpolation model from input model spectra.

    Parameters
    ----------
    inputDir : `str`
        Directory name in which input spectra are stored.
    inputList : `str`
        Ascii table with parameters for interpolation.

        The comma-separated table should contain the following columns:

        - `ModelName` : Spectrum file name. Spectrum should be in FITS format.
        - `Teff` : Effective temperature in K.
        - `Logg` : Surface gravity in log(g/(cm s^-2)).
        - `Z` : Metallicity in [Fe/H].
        - `Alpha` : Alpha element index in [alpha/Fe].

    outputModel : `str`
        RBF model file name to be saved.
    n_procs : `int`
        Number of processes.
    """
    inputModelList = ascii.read(inputList).to_pandas()

    # fetch columns used for making a model
    inputParameter = np.array(inputModelList[["Teff", "Logg", "Z", "Alpha"]])

    # make an empty array for input fluxes
    with fits.open(inputDir + inputModelList["ModelName"].reset_index(drop=True)[0]) as dummyFITS:
        dummyModel = Table(dummyFITS[1].data).to_pandas()
    inputFlux = np.empty((len(dummyModel), len(inputModelList["ModelName"].values)), dtype="float64")

    # Store fluxes of each model into a single array
    for i, row in enumerate(inputModelList["ModelName"].values):
        with fits.open(inputDir + row) as modelFITS:
            model = Table(modelFITS[1].data).to_pandas()
            inputFlux[:, i] = model["Flux"]

    # Function for making an RBF model
    def makeRBFModel(inputflux):
        interpolationFunction = Rbf(inputParameter[:, 0]/1e3,  # Scale down Teff by 1/1000
                                    inputParameter[:, 1],
                                    inputParameter[:, 2],
                                    inputParameter[:, 3],
                                    inputflux,
                                    function="multiquadric",
                                    epsilon=2.0)
        return interpolationFunction

    # Execute `makeRBFModel` in parallel processsing
    interpolationFunction = parallel.parallel_map(makeRBFModel, inputFlux, n_procs)

    # Save the pickle model
    with open(outputModel, "wb") as savePickle:
        pickle.dump(interpolationFunction, savePickle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputDir", help="Directory name of input spectra")
    parser.add_argument("inputList", help="Ascii table with parameters for interpolation")
    parser.add_argument("n_procs", type=int, help="int. Number of processes")
    parser.add_argument("--outputModel", default="ambreRBF.pickle", help="Output model file name")
    args = parser.parse_args()

    makeRBFInterpolationModel(args.inputDir, args.inputList, args.outputModel, args.n_procs)
