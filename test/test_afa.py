from autodp.mechanism_zoo import GaussianMechanism, RandresponseMechanism, ExactGaussianMechanism, SubSampleGaussian_phi
from autodp.transformer_zoo import Composition, ComposeAFA, ComposeGaussian, AmplificationBySampling
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import time

## Example 1:  RDP composition of a homogenous sequence of mechanisms
sigma1 = 5

gm1 = GaussianMechanism(sigma1, phi_off = False, name='phi_GM1')

compose = ComposeAFA()
composed_mech = compose([gm1], [10])
delta1 = 1e-6
eps1 = composed_mech.get_approxDP(delta1)

# RDP-based accountant.
gm2 = GaussianMechanism(sigma1, name='rdp_GM2')
compose_rdp = Composition()
composed_mech_rdp = compose_rdp([gm2], [10])

#Exact Gaussian mechanism.
gm3 = ExactGaussianMechanism(sigma1, name = 'exact_GM3')
compose_exact = ComposeGaussian()
composed_mech_exact = compose_exact([gm3], [10])

# Get name of the composed object, a structured description of the mechanism generated automatically
print('Mechanism name is \"', composed_mech.name,'\"')
print('Parameters are: ',composed_mech.params)
print('epsilon(delta) = ', eps1, ', at delta = ', delta1)
print('Results from rdp_based accountant, epsilon(delta) = ', composed_mech_rdp.get_approxDP(delta1), ', at delta = ', delta1)
print('Results from AFA, epsilon(delta) = ', eps1, ', at delta = ', delta1)
print('Results from exact Gaussian accountant, epsilon(delta) = ', composed_mech_exact.get_approxDP(delta1), ', at delta = ', delta1)

## Example 2:  Composition of a heterogeneous sequence of mechanisms [Gaussian mechanism, randomized response ...].
"""
Consider compositions of Gaussian mechanism with sensitivity 1 and Randomized Response mechanism with probability p.
Consider sigma = 5.0, p =0.52, epsilon =2.0 and we compare delta(epsilon).
The composition looks like [ Gaussian, RR, Gaussian, RR ...] 

prob = 0.52
eps = 2.0
delta = 1e-6
sigma = 5.0
rr_phi = RandresponseMechanism(p=prob, phi_off=False)
rr_rdp = RandresponseMechanism(p=prob)

gm1 = GaussianMechanism(sigma, phi_off = False, name='phi_GM1')
gm2 = GaussianMechanism(sigma, name='rdp_GM2')
compose_afa = ComposeAFA()
compose_rdp = Composition()
compose_len = [int(x**1.25) for x in range(10)]
eps_rdp = []
eps_phi = []
cur_composed_phi = compose_afa([gm1, rr_phi], [1, 1])
cur_composed_rdp = compose_rdp([gm2, rr_rdp], [1, 1])


for i in compose_len:
    print('i', i)
    eps_rdp.append(cur_composed_rdp.approxDP(delta))
    eps_phi.append(cur_composed_phi.approxDP(delta))
    cur_composed_phi = compose_afa([ gm1, rr_phi], [i, i])
    cur_composed_rdp = compose_rdp([ gm2, rr_rdp], [i, i])



plt.figure(figsize = (6,6))

plt.plot(compose_len, eps_rdp , 'm', linewidth=2)
plt.plot(compose_len, eps_phi, 'D--', color = 'pink', linewidth=2)

plt.yscale('log')
plt.legend(
    [r'RDP ','Our AFA'], loc='best', fontsize=22)
plt.grid(True)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel(r'Number of Compositions $k$', fontsize=22)
plt.ylabel(r'$\epsilon$', fontsize=22)

plt.show()
"""
"""
#Example 3, the composition of Poisson Subsampled Gaussian mechanisms.
 #Privacy amplification by sampling. Consider the composition of Poisson subsampled Gaussian mechanisms.
 #The sampling probability is prob=0.02. Our AFA provides the valid lower and upper bounds of privacy cost
 #over composition.
"""
sigma = 2.0
delta = 1e-6
prob = 0.02
eps_rdp = []
# the lower bound of privacy cost over composition.
eps_phi_lower = []
eps_phi_upper = []
#eps_quadrature = []
# klist is the list of #compositions. We consider composition ranges from 200 to 1600.
klist = [100 * i for i in range(2, 5)]
gm1 = GaussianMechanism(sigma, name='GM1')
# The RDP-based amplificationBySampling
poisson_sample = AmplificationBySampling(PoissonSampling=True)
# The AFA tracts the lower and the upper bounds of privacy cost.
phi_subsample_lower = SubSampleGaussian_phi(sigma, prob, lower_bound=True)
phi_subsample_upper = SubSampleGaussian_phi(sigma, prob, upper_bound=True)
compose_afa = ComposeAFA()
compose_rdp = Composition()
eps = 2.
for coeff in klist:

    rdp_composed_mech = compose_rdp([poisson_sample(gm1, prob, improved_bound_flag=True)], [coeff])
    eps_rdp.append(rdp_composed_mech.approx_delta(eps))
    print('eps_rdp', eps_rdp)
    t1 = time.time()
    afa_composed_mech_lower = compose_afa([phi_subsample_lower], [coeff])

    afa_composed_mech_upper = compose_afa([phi_subsample_upper], [coeff])
    eps_phi_lower.append(afa_composed_mech_lower.approx_delta(eps))
    t2 = time.time()
    print('composition time at coeff', coeff, 'is', t2 - t1)
    print('eps_lower', eps_phi_lower)
    eps_phi_upper.append(afa_composed_mech_upper.approx_delta(eps))
    print('eps_upper', eps_phi_upper)
    #eps_quarture.append(phi_quarture.get_approxDP(delta))

plt.figure(figsize = (6,6))

plt.plot(klist, eps_rdp , 'm', linewidth=2)
plt.plot(klist, eps_phi_lower, 'cx-', linewidth=2)
plt.plot(klist, eps_phi_upper, 'rx-', linewidth=2)
plt.yscale('log')
plt.legend(
    [r'RDP with $\epsilon=2.0$','AFA lower bound with $\epsilon=2.0$', 'AFA upper bound with $\epsilon=2.0$'], loc='best', fontsize=22)
plt.grid(True)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel(r'Number of Compositions $k$', fontsize=22)
plt.ylabel(r'$\delta$', fontsize=22)

plt.show()
