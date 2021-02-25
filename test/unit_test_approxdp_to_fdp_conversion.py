from autodp.mechanism_zoo import GaussianMechanism
from autodp.fdp_bank import fDP_gaussian

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

params = [0.05, 0.1, 0.2, 0.5,1.0, 2.0,5.0, 10.0]


def _fdp_conversion(sigma):

    # Using the log(1-f(fpr))  and log(- \partial f(fpr)) that are implemented dedicatedly

    fpr_list = np.linspace(0, 1, 21)

    #  analytical gaussian implementation  (privacy profile)
    gm2 = GaussianMechanism(sigma, name='GM2', RDP_off=True)

    # direct f-DP implementation
    fdp = lambda x: fDP_gaussian({'sigma': sigma},x)

    fdp_direct = fdp(fpr_list)

    # the fdp is converted by numerical methods from privacy profile.
    fdp_converted = np.array([gm2.get_fDP(fpr) for fpr in fpr_list])

    return fdp_direct - fdp_converted



class Test_approxDP2fDP_Conversion(parameterized.TestCase):

    @parameterized.parameters(p for p in params)
    def test_fdp_conversion(self, sigma):
        max_diff = _fdp_conversion(sigma)
        self.assertSequenceAlmostEqual(max_diff, np.zeros_like(max_diff), places=4)


if __name__ == '__main__':
    absltest.main()

