from autodp.mechanism_zoo import ExponentialMechanism, RandresponseMechanism, ExactGaussianMechanism, SubSampleGaussian_phi
from autodp.transformer_zoo import Composition, ComposeAFA, ComposeGaussian, AmplificationBySampling
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


## Example 1:  RDP composition of a homogenous sequence of mechanisms
eps = 1

exp1 = ExponentialMechanism(eps, name= 'exp_cdp')
exp2 = ExponentialMechanism(eps, phi_off = False, name='phi_exp')

print('prob', np.exp(1)/(np.exp(1)+eps))
prob = np.exp(1)/(np.exp(1)+eps)
exp3 = RandresponseMechanism(p=prob, phi_off=False)
delta1 = 1e-6
eps1 = exp1.get_approxDP(delta1)
eps2 = exp2.get_approxDP(delta1)
print('zcdp gives', eps1, 'phi-function gives', eps2)
print('randmized response', exp3.get_approxDP(delta1))

"""
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



plt.figure(figsize = (6,6))

plt.plot(range(compose_len), eps_rdp , 'm', linewidth=2)
plt.plot(range(compose_len), eps_phi, 'D--', color = 'pink', linewidth=2)

plt.yscale('log')
plt.legend(
    [r'RDP  $\epsilon=2.0$','Our AFA $\epsilon=2.0$'], loc='best', fontsize=22)
plt.grid(True)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel(r'Number of Compositions $k$', fontsize=22)
plt.ylabel(r'$\delta$', fontsize=22)

plt.show()


#Example 3, the composition of Poisson Subsampled Gaussian mechanisms.
 #Privacy amplification by sampling. Consider the composition of Poisson subsampled Gaussian mechanisms.
 #The sampling probability is prob=0.02. Our AFA provides the valid lower and upper bounds of privacy cost
 #over composition.

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
for coeff in klist:

    rdp_composed_mech = compose_rdp([poisson_sample(gm1, prob, improved_bound_flag=True)], [coeff])
    eps_rdp.append(rdp_composed_mech.approxDP(delta))
    print('eps_rdp', eps_rdp)
    afa_composed_mech_lower = compose_afa([phi_subsample_lower], [coeff])
    afa_composed_mech_upper = compose_afa([phi_subsample_upper], [coeff])
    eps_phi_lower.append(afa_composed_mech_lower.approxDP(delta))
    print('eps_lower', eps_phi_lower)
    eps_phi_upper.append(afa_composed_mech_upper.approxDP(delta))
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
"""