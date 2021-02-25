from autodp.mechanism_zoo import GaussianMechanism
from autodp.dp_bank import get_eps_ana_gaussian

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

params = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def _fdp_conversion(sigma):

    delta_list = [0,1e-8, 1e-6, 1e-4, 1e-2, 0.3, 0.5, 1]

    # f-DP implementation
    gm3 = GaussianMechanism(sigma, name='GM3', RDP_off=True, approxDP_off=True, fdp_off=False)

    # direct approxdp implementation
    agm = lambda x: get_eps_ana_gaussian(sigma, x)

    eps_direct = np.array([agm(delta) for delta in delta_list])

    # the fdp is converted by numerical methods from privacy profile.
    eps_converted = np.array([gm3.get_approxDP(delta) for delta in delta_list])
    max_diff = eps_direct - eps_converted

    rel_diff = max_diff / (eps_direct+1e-10)

    if np.isinf(eps_direct[0]) and np.isinf(eps_converted[0]):
        rel_diff[0] = 0
    return rel_diff


_fdp_conversion(1.0)

class Test_approxDP2fDP_Conversion(parameterized.TestCase):

    @parameterized.parameters(p for p in params)
    def test_fdp_conversion(self, sigma):
        max_diff = _fdp_conversion(sigma)
        self.assertSequenceAlmostEqual(max_diff, np.zeros_like(max_diff), places=2)


if __name__ == '__main__':
    absltest.main()

