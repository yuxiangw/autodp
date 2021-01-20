import numpy as np
from autodp.autodp.mechanism_zoo import GaussianMechanism, PureDP_Mechanism

# Example 1: Gaussian mechanism

sigma = 2.0


gm0 = GaussianMechanism(sigma,name='GM0',approxDP_off=True, use_basic_RDP_to_approxDP_conversion=True)
gm1 = GaussianMechanism(sigma,name='GM1',approxDP_off=True)
gm1b = GaussianMechanism(sigma,name='GM1b',approxDP_off=True, use_fDP_based_RDP_to_approxDP_conversion=True)
gm2 = GaussianMechanism(sigma,name='GM2',RDP_off=True)
gm3 = GaussianMechanism(sigma,name='GM3',RDP_off=True, approxDP_off=True, fdp_off=False)



eps = np.sqrt(2)/sigma # Aligning the variance of the laplace mech and gaussian mech
laplace = PureDP_Mechanism(eps,name='Laplace')

label_list = ['naive_RDP_conversion','BBGHS_RDP_conversion','Our new method',
              'exact_eps_delta_DP','exact_fdp',r'laplace mech ($b = \sqrt{2}/\sigma$)']


import matplotlib.pyplot as plt



fpr_list, fnr_list = gm0.plot_fDP()
fpr_list1, fnr_list1 = gm1.plot_fDP()
fpr_list1b, fnr_list1b = gm1b.plot_fDP()
fpr_list2, fnr_list2 = gm2.plot_fDP()
fpr_list3, fnr_list3 = gm3.plot_fDP()
fpr_list4, fnr_list4 = laplace.plot_fDP()

plt.figure(figsize=(4,4))
plt.plot(fpr_list,fnr_list)
plt.plot(fpr_list1,fnr_list1)
plt.plot(fpr_list1b,fnr_list1b)
plt.plot(fpr_list2, fnr_list2)
plt.plot(fpr_list3, fnr_list3,':')
plt.plot(fpr_list4, fnr_list4,'-.')
plt.legend(label_list)
plt.xlabel('Type I error')
plt.ylabel('Type II error')
plt.savefig('rdp2fdp.pdf')
plt.show()



delta = 1e-3


eps3 = gm3.approxDP(delta)
eps0 = gm0.approxDP(delta)
eps1 = gm1.approxDP(delta)
eps1b = gm1b.approxDP(delta)

eps2 = gm2.approxDP(delta)

eps4 = laplace.approxDP(delta)

epsilons = [eps0,eps1,eps1b,eps2,eps3,eps4]

print(epsilons)

plt.bar(label_list,epsilons)
plt.xticks(rotation=45, ha="right")
plt.show()

