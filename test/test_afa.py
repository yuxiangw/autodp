"""
Example on computing tight privacy guarantees through AFA.
The method is described in https://arxiv.org/pdf/2106.08567.pdf

Example 1: compositions of Gaussian mechansim

"""
from autodp.mechanism_zoo import GaussianMechanism, RandresponseMechanism, ExactGaussianMechanism, SubSampleGaussian_phi
from autodp.transformer_zoo import Composition, ComposeAFA, ComposeGaussian, AmplificationBySampling
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import time

"""
#Example 3, the composition of Poisson Subsampled Gaussian mechanisms.
 #Privacy amplification by sampling. Consider the composition of Poisson subsampled Gaussian mechanisms.
 #The sampling probability is prob=0.02. Our AFA provides the valid lower and upper bounds of privacy cost
 #over composition.
"""

#sigma = 2.0
sigma = 2.0
delta = 1e-5
prob = 0.02
eps_rdp = []
# the lower bound of privacy cost over composition.
eps_phi_lower = []
eps_phi_upper = []
eps_quad = []
# klist is the list of #compositions. We consider composition ranges from 200 to 1600.
klist = [100* i for i in range(2,16)]
gm1 = GaussianMechanism(sigma, name='GM1')
t0 = time.time()
# The RDP-based amplificationBySampling
poisson_sample = AmplificationBySampling(PoissonSampling=True)
# AFA with double quadrature: applying Gaussian quadrature to calculate the characteristic functions directly when
# the closed form phi functions do not exist.

phi_subsample_quad = SubSampleGaussian_phi(sigma, prob)
compose_afa = ComposeAFA()
compose_rdp = Composition()

print('pre time', time.time()-t0)
# The lower bound from PLD-accountant https://arxiv.org/abs/2102.12412
fft_lower = [0.5738987073827246,0.7070330939276875,0.8213662927173179,0.92358900507979, 1.0171625920742302,1.1041565229625558,1.1859259173798518, 1.2634158671679698, 1.3373168083154394, 1.4081514706603082, 1.4763269642671626, 1.54216775460533, 1.605937390055278,1.667853467197358]
fft_higher = [0.5738987073827246,0.7070330939276875, 0.8213662927173179, 0.92358900507979, 1.0171625920742302, 1.1041565229625558, 1.1859259173798518,1.2634158671679698,1.3373168083154394, 1.4081514706603082, 1.4763269642671626, 1.54216775460533, 1.605937390055278, 1.667853467197358]
for coeff in klist:
    t2 = time.time()
    # RDP-based accountant with a tighter RDP to (epsilon, delta)-DP conversion.
    rdp_composed_mech = compose_rdp([poisson_sample(gm1, prob, improved_bound_flag=True)], [coeff])
    eps_rdp.append(rdp_composed_mech.approxDP(delta))
    print('eps_rdp', eps_rdp, 'time for rdp', time.time()-t2)
    afa_phi_quad = compose_afa([phi_subsample_quad], [coeff])
    eps_quad.append(afa_phi_quad.approxDP(delta))
    print('quadrature', eps_quad)




plt.figure(figsize = (6,6))

plt.plot(klist, eps_rdp , 'm.-.', linewidth=2)
plt.plot(klist, eps_quad, 'bx-', linewidth=2)
plt.plot(klist, fft_lower, 'gs-', linewidth=2)
plt.plot(klist, fft_upper, 'rs-', linewidth=2)




#plt.legend(
#    [r'RDP','AFA with Double quadrature', 'FA lower bound', 'FA upper bound'], loc='best', fontsize=22)

plt.grid(True)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel(r'Number of Compositions $k$', fontsize=22)
plt.ylabel(r'$\delta$', fontsize=22)

plt.show()
