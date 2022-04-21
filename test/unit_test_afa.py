from autodp.mechanism_zoo import GaussianMechanism, ExactGaussianMechanism
from autodp.fdp_bank import fDP_gaussian

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

params = [0.05, 0.1, 0.2, 0.5,1.0, 2.0,5.0, 10.0]
delta_list = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]

def _test_diff(sigma):


    diff = []


    for delta in delta_list:
        # Define phi function based Gaussian mechanism.
        gm1 = GaussianMechanism(sigma, phi_off=False, name='phi_GM1')

        # Exact Gaussian mechanism.
        gm3 = ExactGaussianMechanism(sigma, name='exact_GM3')

        diff.append(gm1.get_approxDP(delta) - gm3.get_approxDP(delta))
        print('diff', diff)
    return np.array(diff)


_test_diff(2.0)

class Test_approxDP2fDP_Conversion(parameterized.TestCase):

    @parameterized.parameters(p for p in params)
    def test_fdp_conversion(self, sigma):
        max_diff = _test_diff(sigma)
        print('max_diff', max_diff)
        self.assertSequenceAlmostEqual(max_diff, np.zeros_like(max_diff), places=4)


if __name__ == '__main__':
    absltest.main()

