from autodp.mechanism_zoo import GaussianMechanism
from autodp.dp_bank import get_eps_ana_gaussian
from autodp.phi_bank import phi_subsample_gaussian_q, phi_subsample_gaussian_p, phi_gaussian
from autodp.transformer_zoo import AmplificationBySampling_pld
import scipy.integrate as integrate
import numpy as np
import math
from scipy.stats import norm
from absl.testing import absltest
from absl.testing import parameterized

params = [0.1]


def _pld_sample_remove_only(sigma):

    t_list = [1.1**x for x in range(2, 20)]
    # Phi-function implementation
    gm = GaussianMechanism(sigma, name='GM3', RDP_off=True, approxDP_off=True, phi_off=False)

    # sampling ratio
    gamma = 0.01
    # direct compute the phi function of subsample gaussian for "removal only" neighboring relationship
    phi_p2q = lambda x: phi_subsample_gaussian_p({'sigma':sigma, 'gamma':gamma}, x, remove_only=True)
    phi_q2p = lambda x: phi_subsample_gaussian_q({'sigma':sigma, 'gamma':gamma}, x,remove_only=True)
    sample_remove_only = AmplificationBySampling_pld(PoissonSampling=True, neighboring='remove_only')
    sample_gau = sample_remove_only(gm, gamma)
    sample_phi_p2q = lambda x: sample_gau.log_phi_p2q(x)
    sample_phi_q2p = lambda x: sample_gau.log_phi_q2p(x)
    phi_direct_p2q = np.array([phi_p2q(t) for t in t_list])
    phi_direct_q2p = np.array([phi_q2p(t) for t in t_list])

    # the phi function obtained through amplification by sampling transformer
    phi_converted_p2q = np.array([sample_phi_p2q(t) for t in t_list])
    phi_converted_q2p =np.array([sample_phi_q2p(t) for t in t_list])
    max_diff_p2q = phi_direct_p2q - phi_converted_p2q
    max_diff_q2p = phi_direct_q2p - phi_converted_q2p
    rel_diff_p2q = max_diff_p2q / (phi_direct_p2q + 1e-15)
    rel_diff_q2p = max_diff_q2p / (phi_direct_q2p + 1e-15)

    if np.isinf(phi_direct_p2q[0]) and np.isinf(phi_converted_p2q[0]):
        rel_diff_p2q[0] = 0
    if np.isinf(phi_direct_q2p[0]) and np.isinf(phi_converted_q2p[0]):
        rel_diff_q2p[0] = 0
    ref_diff = [max(diff_p2q, diff_q2p) for (diff_p2q, diff_q2p) in zip(rel_diff_p2q, rel_diff_q2p)]
    return ref_diff

def _pld_sample_add_only(sigma):

    t_list = [1.1**x for x in range(2, 20)]
    # Phi-function implementation
    gm = GaussianMechanism(sigma, name='GM3', RDP_off=True, approxDP_off=True, phi_off=False)
    # sampling ratio
    gamma = 0.001
    # direct compute the phi function of subsample gaussian for "removal only" neighboring relationship
    phi_p = lambda x: phi_subsample_gaussian_p({'sigma':sigma, 'gamma':gamma}, x, remove_only=False)
    phi_direct = np.array([phi_p(t) for t in t_list])
    sample_add_only = AmplificationBySampling_pld(PoissonSampling=True, neighboring='add_only')
    sample_gau = sample_add_only(gm, gamma)
    # when we update pdf, we shall update phi
    sample_phi_p2q = lambda x: sample_gau.log_phi_p2q(x)

    # the phi function obtained through amplification by sampling transformer
    phi_converted = np.array([sample_phi_p2q(t) for t in t_list])
    max_diff = phi_direct - phi_converted

    rel_diff = max_diff / (phi_direct+1e-10)

    if np.isinf(phi_direct[0]) and np.isinf(phi_converted[0]):
        rel_diff[0] = 0

    return rel_diff

#_pld_sample(1.)

class Test_approxDP2fDP_Conversion(parameterized.TestCase):


    @parameterized.parameters(p for p in params)
    def test_pld_sample_remove_only(self, sigma):
        max_diff = _pld_sample_remove_only(sigma)
        self.assertSequenceAlmostEqual(max_diff, np.zeros_like(max_diff), places=2)

    @parameterized.parameters(p for p in params)
    def test_pld_sample_add_only(self, sigma):
        max_diff = _pld_sample_add_only(sigma)
        self.assertSequenceAlmostEqual(max_diff, np.zeros_like(max_diff), places=2)


if __name__ == '__main__':
    absltest.main()

