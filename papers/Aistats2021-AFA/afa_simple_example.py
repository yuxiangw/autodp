"""
Example on computing tight privacy guarantees through AFA.
Example 1: composition of Gaussian mechanism
Example 2: composition of a heterogeneous sequence of mechanisms


The workflow of AFA:
Step 1. Describe each mechanism using a pair of characteristic functions (phi functions).
Step 2: Define an accountant to track the log of phi-functions.
Step 3. For delta(eps) or eps(delta) queries, we will use numerical inversion to
convert the characteristic function back to CDFs (see cdf_bank.py). Then using an equivalent definition of DP
to convert the CDF result to the DP guarantee (see converter.py).


The detailed method is described in https://arxiv.org/pdf/2106.08567.pdf


"""
from autodp.mechanism_zoo import GaussianMechanism, RandresponseMechanism, ExactGaussianMechanism, SubSampleGaussian_phi
from autodp.transformer_zoo import Composition, ComposeAFA, ComposeGaussian, AmplificationBySampling
import matplotlib.pyplot as plt
import numpy as np
import time

## Example 1:  Composition on Gaussian mechanisms.
doc = {}
delta = 1e-6
# the number of compositions
klist = [80 * i for i in range(2, 15)][3:]
for sigma in [30, 60]:
    # epsilon computed through RDP
    eps_rdp = []
    # Analytical Gaussian
    eps_exact = []
    # phi function
    eps_phi = []
    # Declare a mechanism with a characteristic function-based description.
    gm1 = GaussianMechanism(sigma, phi_off=False, name='phi_GM1')
    # Define an analytical Fourier Accountant(AFA).
    compose = ComposeAFA()

    # RDP-based accountant.
    gm2 = GaussianMechanism(sigma, name='rdp_GM2')
    compose_rdp = Composition()


    # Exact Gaussian mechanism.
    gm3 = ExactGaussianMechanism(sigma, name='exact_GM3')
    compose_exact = ComposeGaussian()

    for coeff in klist:

        composed_mech_afa = compose([gm1], [coeff])
        eps_afa = composed_mech_afa.get_approxDP(delta)
        # Get name of the composed object, a structured description of the mechanism generated automatically
        # print('Mechanism name is \"', composed_mech.name, '\"')
        # print('Parameters are: ', composed_mech.params)
        composed_mech_rdp = compose_rdp([gm2], [coeff])
        composed_mech_exact = compose_exact([gm3], [coeff])

        eps_rdp.append(composed_mech_rdp.get_approxDP(delta))
        eps_exact.append(composed_mech_exact.get_approxDP(delta))
        eps_phi.append(eps_afa)

    cur_result = {}
    cur_result['rdp'] = eps_rdp
    cur_result['gt'] = eps_exact
    cur_result['phi'] = eps_phi
    doc[str(sigma)] = cur_result
# with open(path, 'wb') as f:
#    pickle.dump(doc, f)


import matplotlib.pyplot as plt


plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(klist, doc['30']['rdp'], 'm', linewidth=2)
plt.plot(klist, doc['30']['gt'], 'D--', color='pink', linewidth=2)
plt.plot(klist, doc['30']['phi'], 'x-', color='darkred', linewidth=2)
plt.plot(klist, doc['60']['rdp'], color='darkorange', linewidth=2)
plt.plot(klist, doc['60']['gt'], 'D--', color='pink', linewidth=2)
plt.plot(klist, doc['60']['phi'], 'x-', color='darkblue', linewidth=2)
plt.yscale('log')
plt.legend(
    [r'RDP $\sigma=30$', 'Exact Accountant $\sigma=30$', 'Our AFA $\sigma=30$', 'RDP $\sigma=60$',
     'Exact Accountant $\sigma=60$', 'Our AFA with $\sigma=60$'], loc='best', fontsize=18)
plt.grid(True)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel(r'Number of Compositions $k$', fontsize=22)
plt.ylabel(r'$\epsilon$', fontsize=22)
plt.show()


# Example 2: Composition of a heterogeneous sequence of mechanisms [Gaussian mechanism, randomized response ...].
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
compose_len = [int(x*50) for x in range(20)]
eps_rdp = []
eps_phi = []
cur_composed_phi = compose_afa([gm1, rr_phi], [1, 1])
cur_composed_rdp = compose_rdp([gm2, rr_rdp], [1, 1])


for i in compose_len:
    eps_rdp.append(cur_composed_rdp.approxDP(delta))
    eps_phi.append(cur_composed_phi.approxDP(delta))
    cur_composed_phi = compose_afa([ gm1, rr_phi], [i, i])
    cur_composed_rdp = compose_rdp([ gm2, rr_rdp], [i, i])



plt.figure(figsize = (6,6))

plt.plot(compose_len, eps_rdp , 'm', linewidth=2)
plt.plot(compose_len, eps_phi, 'D--', color = 'pink', linewidth=2)

plt.legend(
    [r'RDP ','Our AFA'], loc='best', fontsize=22)
plt.grid(True)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel(r'Number of Compositions $k$', fontsize=22)
plt.ylabel(r'$\epsilon$', fontsize=22)

plt.show()

