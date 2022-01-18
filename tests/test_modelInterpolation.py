from pfs.drp.fstar.modelInterpolation import ModelInterpolation
import lsst.utils
import lsst.utils.tests

import os


class ModelInterpolationTestCase(lsst.utils.tests.TestCase):
    def test(self):
        try:
            dataDir = lsst.utils.getPackageDir("fluxmodeldata")
        except LookupError:
            self.skipTest("fluxmodeldata not setup")

        if not os.path.exists(os.path.join(dataDir, "interpolator.pickle")):
            self.skipTest("makeRBFInterpolationModel.py has not been run")

        model = ModelInterpolation.fromFluxModelData(dataDir)
        wavelength, flux = model.interpolate(teff=7777, logg=3.333, metal=0.555, alpha=0.222)
        self.assertEqual(len(wavelength.shape), 1)
        self.assertEqual(wavelength.shape, flux.shape)


def setup_module(module):
    lsst.utils.tests.init()
