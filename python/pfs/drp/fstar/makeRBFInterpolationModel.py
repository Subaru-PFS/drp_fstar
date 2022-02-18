import lsst.utils

import numpy as np
from astropy.io import fits
from scipy.interpolate import Rbf

import argparse
import os
import pickle

from . import parallel


def makeRBFInterpolationModel(fluxmodeldataPath, nProcs=1):
    """Generate an RBF interpolation model from input model spectra.

    Parameters
    ----------
    fluxmodeldataPath : `str`
        Path to ``fluxmodeldata`` package.
    nProcs : `int`
        Number of processes.
    """
    with fits.open(os.path.join(fluxmodeldataPath, "broadband", "photometries.fits")) as hdus:
        inputParameters = np.asarray(hdus[1].data)

    # make an empty array for input fluxes
    param = inputParameters[0]
    flux, wcs = _loadFluxArray(
        fluxmodeldataPath, param["Teff"], param["Logg"], param["M"], param["Alpha"]
    )
    inputFlux = np.empty((len(flux), len(inputParameters)), dtype="float64")

    # Store fluxes of each model into a single array
    for i, param in enumerate(inputParameters):
        flux, wcs = _loadFluxArray(
            fluxmodeldataPath, param["Teff"], param["Logg"], param["M"], param["Alpha"]
        )
        inputFlux[:, i] = flux

    teff = np.asarray(inputParameters["Teff"], dtype=float) / 1e3
    logg = np.asarray(inputParameters["Logg"], dtype=float)
    m = np.asarray(inputParameters["M"], dtype=float)
    alpha = np.asarray(inputParameters["Alpha"], dtype=float)

    # Function for making an RBF model
    def makeRBFModel(flux):
        return Rbf(
            teff,
            logg,
            m,
            alpha,
            flux,
            function="multiquadric",
            epsilon=2.0,
        )

    # Execute `makeRBFModel` in parallel
    interpolationFunction = parallel.parallel_map(makeRBFModel, inputFlux, nProcs)

    # Pickle the model
    output = os.path.join(fluxmodeldataPath, "interpolator.pickle")
    with open(output, "wb") as f:
        pickle.dump({
            "wcs": wcs,
            "interpolationFunction": interpolationFunction,
        }, f)


def _loadFluxArray(fluxmodeldataPath, teff, logg, m, alpha):
    """Read the flux array from a spectrum file.

    Parameters
    ----------
    fluxmodeldataPath : `str`
        Path to ``fluxmodeldata`` package.
    teff : `float`
        Effective temperature in K.
    logg : `float`
        Surface gravity in log(g/(cm s^-2)).
    m : `float`
        Metallicity in [Fe/H].
    alpha : `float`
        Alpha element index in [alpha/Fe].

    Returns
    -------
    flux : `numpy.array`
        Fluxes at sampling points.
    wcs : `dict`
        FITS headers for converting the unit of wavelength to nm.
    """
    args = {
        "teff": int(round(teff)),
        "logg": logg + 0.0,  # "+ 0.0" turns -0 to +0.
        "m": m + 0.0,
        "alpha": alpha + 0.0,
    }
    path = os.path.join(
        fluxmodeldataPath, "spectra", "fluxmodel_%(teff)d_g_%(logg).2f_z_%(m).2f_a_%(alpha).1f.fits" % args
    )
    with fits.open(path) as hdus:
        header = hdus[0].header
        wcs = {key: header[key] for key in ["CRPIX1", "CDELT1", "CRVAL1"]}
        return hdus[0].data, wcs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fluxmodeldata", nargs="?", help="Path to fluxmodeldata package")
    parser.add_argument("-j", "--nprocs", type=int, default=1, help="Number of processes")
    args = parser.parse_args()

    if not args.fluxmodeldata:
        args.fluxmodeldata = lsst.utils.getPackageDir("fluxmodeldata")

    makeRBFInterpolationModel(args.fluxmodeldata, args.nprocs)
