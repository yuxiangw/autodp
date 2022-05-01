"""
Example on computing tight privacy guarantees through AFA.
The method is described in https://arxiv.org/pdf/2106.08567.pdf

Example 1: compositions of Gaussian mechansim

"""
from autodp.mechanism_zoo import GaussianMechanism
from autodp.transformer_zoo import Composition, ComposeAFA,  AmplificationBySampling
import matplotlib.pyplot as plt
from autodp.transformer_zoo import AmplificationBySampling_pld
import matplotlib.font_manager as fm
import numpy as np
import time

"""
#Example 3, the composition of Poisson Subsampled Gaussian mechanisms.
 #Privacy amplification by sampling. Consider the composition of Poisson subsampled Gaussian mechanisms.
 #The sampling probability is prob=0.02. Our AFA provides the valid lower and upper bounds of privacy cost
 #over composition.
"""


sigma = 2.0
delta = 1e-5
prob = 0.02
eps_rdp = []

# We treat the privacy amplification by sampling rule for add/remove neighboring relation separately.
# We can obtain the results for the standard add/remove by a pointwise maximum of the two.
eps_afa_add_only = []
eps_afa_remove_only = []
# klist is the list of #compositions. We consider composition ranges from 200 to 1600.
klist = [100 * i for i in range(2,16)]
gm1 = GaussianMechanism(sigma, name='GM1')
gm2 = GaussianMechanism(sigma, phi_off=False, name='GM2', RDP_off = True, approxDP_off=True)
gm3 = GaussianMechanism(sigma, phi_off=False, name='GM3', RDP_off = True, approxDP_off=True)
t0 = time.time()
# The RDP-based amplificationBySampling
poisson_sample_rdp = AmplificationBySampling(PoissonSampling=True)
# AFA with double quadrature: applying Gaussian quadrature to calculate the characteristic functions directly when
# the closed form phi functions do not exist.
compose_afa = ComposeAFA()
compose_rdp = Composition()
transformer_remove_only = AmplificationBySampling_pld(PoissonSampling=True, neighboring='remove_only')
transformer_add_only = AmplificationBySampling_pld(PoissonSampling=True, neighboring='add_only')
sample_gau_remove_only =transformer_remove_only(gm2, prob)
sample_gau_add_only =transformer_add_only(gm2, prob)

print('pre time', time.time()-t0)
# The lower bound from PLD-accountant https://arxiv.org/abs/2102.12412
fft_lower = [0.5738987073827246,0.7070330939276875,0.8213662927173179,0.92358900507979, 1.0171625920742302,1.1041565229625558,1.1859259173798518, 1.2634158671679698, 1.3373168083154394, 1.4081514706603082, 1.4763269642671626, 1.54216775460533, 1.605937390055278,1.667853467197358]
fft_higher = [0.5738987073827246,0.7070330939276875, 0.8213662927173179, 0.92358900507979, 1.0171625920742302, 1.1041565229625558, 1.1859259173798518,1.2634158671679698,1.3373168083154394, 1.4081514706603082, 1.4763269642671626, 1.54216775460533, 1.605937390055278, 1.667853467197358]
for coeff in klist:
    t2 = time.time()
    # RDP-based accountant with a tighter RDP to (epsilon, delta)-DP conversion.
    rdp_composed_mech = compose_rdp([poisson_sample_rdp(gm1, prob, improved_bound_flag=True)], [coeff])
    eps_rdp.append(rdp_composed_mech.approxDP(delta))
    print('eps_rdp', eps_rdp, 'time for rdp', time.time()-t2)
    afa_phi_add = compose_afa([sample_gau_add_only], [coeff])
    afa_phi_remove = compose_afa([sample_gau_remove_only], [coeff])
    eps_afa_remove_only.append(afa_phi_remove.approxDP(delta))
    eps_afa_add_only.append(afa_phi_add.approxDP(delta))

    #afa_phi_quad = compose_afa([sample_gau], [coeff])

    print('afa add only', eps_afa_add_only)
    print('afa remove only', eps_afa_remove_only)




plt.figure(figsize = (6,6))

plt.plot(klist, eps_rdp , 'm.-.', linewidth=2)
plt.plot(klist, eps_afa_remove_only, 'bx-', linewidth=2)
plt.plot(klist, eps_afa_add_only, 'bx-', linewidth=2)
plt.plot(klist, fft_lower, 'gs-', linewidth=2)
plt.plot(klist, fft_higher, 'rs-', linewidth=2)




#plt.legend(
#    [r'RDP','AFA with Double quadrature', 'FA lower bound', 'FA upper bound'], loc='best', fontsize=22)

plt.grid(True)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel(r'Number of Compositions $k$', fontsize=22)
plt.ylabel(r'$\delta$', fontsize=22)

plt.show()
