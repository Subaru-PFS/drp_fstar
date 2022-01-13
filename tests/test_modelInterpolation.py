from pfs.drp.fstar.modelInterpolation import ModelInterpolation
import lsst.utils.tests


class ModelInterpolationTestCase(lsst.utils.tests.TestCase):
    def test(self):
        # Tests must be done...
        assert(ModelInterpolation is not None)


def setup_module(module):
    lsst.utils.tests.init()
