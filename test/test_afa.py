from autodp.mechanism_zoo import GaussianMechanism, RandresponseMechanism, ExactGaussianMechanism
from autodp.transformer_zoo import Composition, ComposeAFA, ComposeGaussian
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

## Example 1:  RDP composition of a homogenous sequence of mechanisms
sigma1 = 5.0

gm1 = GaussianMechanism(sigma1, phi_off = False, name='phi_GM1')

compose = ComposeAFA()
composed_mech = compose([gm1], [100])
delta1 = 1e-6
eps1 = composed_mech.get_approxDP(delta1)

# RDP-based accountant.
gm2 = GaussianMechanism(sigma1, name='rdp_GM2')
compose_rdp = Composition()
composed_mech_rdp = compose_rdp([gm2], [100])

#Exact Gaussian mechanism.
gm3 = ExactGaussianMechanism(sigma1, name = 'exact_GM3')
compose_exact = ComposeGaussian()
composed_mech_exact = compose_exact([gm3], [100])

# Get name of the composed object, a structured description of the mechanism generated automatically
print('Mechanism name is \"', composed_mech.name,'\"')
print('Parameters are: ',composed_mech.params)
print('epsilon(delta) = ', eps1, ', at delta = ', delta1)
print('Results from rdp_based accountant, epsilon(delta) = ', composed_mech_rdp.get_approxDP(delta1), ', at delta = ', delta1)
print('Results from exact Gaussian accountant, epsilon(delta) = ', composed_mech_exact.get_approxDP(delta1), ', at delta = ', delta1)


## Example 2:  RDP composition of a heterogeneous sequence of mechanisms [Gaussian mechanism, randomized response ...].
"""
Consider compositions of Gaussian mechanism with sensitivity 1 and Randomized Response mechanism with probability p.
Consider sigma = 5.0, p =0.52, epsilon =2.0 and we compare delta(epsilon).
The composition looks like [ Gaussian, RR, Gaussian, RR ...] 
"""
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
compose_len = 20
eps_rdp = []
eps_phi = []
cur_composed_phi = compose_afa([gm1, rr_phi], [1, 1])
cur_composed_rdp = compose_rdp([gm2, rr_rdp], [1, 1])


for i in range(compose_len):
    print('i', i)
    eps_rdp.append(cur_composed_rdp.approxDP(delta))
    eps_phi.append(cur_composed_phi.approxDP(delta))
    cur_composed_phi = compose_afa([cur_composed_phi, gm1, rr_phi], [1, 1, 1])
    cur_composed_rdp = compose_rdp([cur_composed_rdp, gm2, rr_rdp], [1, 1, 1])



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
