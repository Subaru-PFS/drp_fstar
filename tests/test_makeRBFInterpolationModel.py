from pfs.drp.fstar.makeRBFInterpolationModel import makeRBFInterpolationModel
import lsst.utils.tests


class MakeRBFInterpolationModelTestCase(lsst.utils.tests.TestCase):
    def test(self):
        # Tests must be done...
        assert(makeRBFInterpolationModel is not None)


def setup_module(module):
    lsst.utils.tests.init()
