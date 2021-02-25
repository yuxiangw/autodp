from autodp.mechanism_zoo import PureDP_Mechanism

from autodp.transformer_zoo import Composition

# Example: pure DP mechanism and composition of it

eps = 0.3


mech = PureDP_Mechanism(eps, name='Laplace')

import matplotlib.pyplot as plt

fpr_list, fnr_list = mech.plot_fDP()
plt.figure(1)
plt.plot(fpr_list,fnr_list,label='fdp_of_laplace')


delta = 1e-6
epslist = [mech.get_approxDP(delta)]

# declare a transformation to handle composition
compose = Composition()

for i in range(2,11):
    mech_composed = compose([mech], [i])
    epslist.append(mech_composed.get_approxDP(delta))
    fpr_list, fnr_list = mech_composed.plot_fDP()
    plt.plot(fpr_list, fnr_list, label='fdp_of_'+str(i)+'laplace')

plt.legend()
plt.xlabel('Type I error')
plt.ylabel('Type II error')
plt.show()
# we could specify parameters of the composition, e.g. using RDP composition, using KOV and so on



plt.figure(2)
plt.plot(range(1,11),epslist)
plt.xlabel('number of times compose')
plt.ylabel(r'$\epsilon$ at $\delta = 1e-6$')
plt.show()



