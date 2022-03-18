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
import pickle
import sys
sys.path.append('../../test')
dp_fed_20 = 'DP-FedAvg_local_it_20_convergence_lr.pkl'

dp_fed_4 = 'DP-FedAvg_local_it_4_convergence_lr.pkl'
fed_4 = '-FedAvg_local_it_4_convergence_lr.pkl'
fed_20 = '-FedAvg_local_it_20_convergence_lr.pkl'
fed_40 = 'FedAvg_local_it_40_convergence_lr.pkl'
fed_80 = 'FedAvg_local_it_80_convergence_lr.pkl'
dp_fed_40 = 'DP-FedAvg_local_it_40_convergence_lr.pkl'
dp_fed_80 = 'DP-FedAvg_local_it_80_convergence_lr.pkl'
import matplotlib.pyplot as plt

t = 50


f, ax = plt.subplots()
plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
with open(dp_fed_80, 'rb') as f:
    result = pickle.load(f)
c_round = len(result['ac'])
c_round_range = range(t)
plt.plot(c_round_range, result['ac'][:t], 'b.-.', linewidth=2)
with open(dp_fed_40, 'rb') as f:
    result = pickle.load(f)

plt.plot(c_round_range, result['ac'][:t], 'k-', linewidth=2)
with open(dp_fed_20, 'rb') as f:
    result = pickle.load(f)

plt.plot(c_round_range, result['ac'][:t], 'r--', linewidth=2)
#with open(dp_fed_4, 'rb') as f:
#    result = pickle.load(f)
#c_round = len(result['ac'])
#c_round_range = range(c_round)
#plt.plot(c_round_range, result['ac'], 'g.-.', linewidth=2)

with open(fed_80, 'rb') as f:
    result = pickle.load(f)
plt.plot(c_round_range, result['ac'][:t], 'm.-.', linewidth=2)

with open(fed_40, 'rb') as f:
    result = pickle.load(f)
plt.plot(c_round_range, result['ac'][:t], 'g.-', linewidth=2)
with open(fed_20, 'rb') as f:
    result = pickle.load(f)
plt.plot(c_round_range, result['ac'][:t], 'c--', linewidth=2)
#plt.plot(klist, eps_optimal_rdp, 'y.-', linewidth=2)


plt.legend(
    [r'DP-FedAvg_E_80','DP-FedAvg_E_40', 'DP-FedAvg_E_20','FedAvg_E_80', 'FedAvg_E_40', 'FedAvg_E_20'], loc='best', fontsize=17)
#plt.legend(
#    [r'DP-FedAvg_E_80','DP-FedAvg_E_40', 'DP-FedAvg_E_20','DP-FedAvg_E_4', 'FedAvg_E_20', 'FedAvg_E_4'], loc='best', fontsize=17)
plt.grid(True)
plt.ylim([0.55, 0.88])
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel(r'Communication rounds', fontsize=22)
plt.ylabel(r'Accuray', fontsize=22)

#plt.show()
plt.savefig('ac_commu.pdf')
#plt.savefig('exp4_eps.pdf', bbox_inches='tight')



from autodp.mechanism_zoo import GaussianMechanism, RandresponseMechanism, ExactGaussianMechanism, SubSampleGaussian_phi
from autodp.transformer_zoo import Composition, ComposeAFA, ComposeGaussian, AmplificationBySampling
import matplotlib.pyplot as plt
import numpy as np
import time

## Example 1:  Composition on Gaussian mechanisms.
doc = {}

delta = 1e-3
# the number of compositions
coeff = 400
eps_rdp = []
for sigma in [13.5,17, 15, 20, 25, 30, 35, 40]:
    # epsilon computed through RDP



    # RDP-based accountant.
    gm2 = GaussianMechanism(sigma, name='rdp_GM2')
    compose_rdp = Composition()

    composed_mech_rdp = compose_rdp([gm2], [coeff])
    eps_rdp.append(composed_mech_rdp.get_approxDP(delta))




print('cur_result', eps_rdp)



t0 = time.time()
# The RDP-based amplificationBySampling
poisson_sample = AmplificationBySampling(PoissonSampling=True)
# AFA with double quadrature: applying Gaussian quadrature to calculate the characteristic functions directly when
# the closed form phi functions do not exist.

compose_rdp = Composition()
prob = 0.04
noisy_scale = 0.02
eps_rdp = []
clip = 0.03
sigma = noisy_scale / clip
klist = [5*i for i in range(6, 12)]
gm1 = GaussianMechanism(sigma, name='GM1')
for coeff in klist:
    t2 = time.time()
    # RDP-based accountant with a tighter RDP to (epsilon, delta)-DP conversion.
    rdp_composed_mech = compose_rdp([poisson_sample(gm1, prob, improved_bound_flag=True)], [coeff])
    eps_rdp.append(rdp_composed_mech.approxDP(delta))
print('eps_rdp', eps_rdp, 'composition', coeff)


