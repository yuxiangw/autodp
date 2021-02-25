
from autodp.mechanism_zoo import ExactGaussianMechanism
from autodp.transformer_zoo import Composition, ComposeGaussian
import matplotlib.pyplot as plt

sigma1 = 5.0
sigma2 = 8.0

gm1 = ExactGaussianMechanism(sigma1,name='GM1')
gm2 = ExactGaussianMechanism(sigma2,name='GM2')

# run gm1 for 3 rounds
# run gm2 for 5 times

# compose them with the transformation: compose and
rdp_compose = Composition()
rdp_composed_mech = rdp_compose([gm1, gm2], [3, 5])

compose = ComposeGaussian()
composed_mech = compose([gm1, gm2], [3, 5])



# Query for eps given delta
delta1 = 1e-6
eps1 = composed_mech.get_approxDP(delta1)
eps1b = rdp_composed_mech.get_approxDP(delta1)

delta2 = 1e-4
eps2 = composed_mech.get_approxDP(delta2)
eps2b = rdp_composed_mech.get_approxDP(delta2)

# Get name of the composed object, a structured description of the mechanism generated automatically
print('Mechanism name is \"', composed_mech.name,'\"')
print('Parameters are: ',composed_mech.params)

print('Generic composition: epsilon(delta) = ', eps1b, ', at delta = ', delta1)
print('Gaussian composition:  epsilon(delta) = ', eps1, ', at delta = ', delta1)

print('Generic composition: epsilon(delta) = ', eps2b, ', at delta = ', delta2)
print('Gaussian composition:  epsilon(delta) = ', eps2, ', at delta = ', delta2)



# Get hypothesis testing interpretation so we can directly plot it
fpr_list, fnr_list = composed_mech.plot_fDP()
fpr_list, fnr_list2 = rdp_composed_mech.plot_fDP()

plt.figure(figsize = (6,6))
plt.plot(fpr_list,fnr_list,label='Gaussian_composition')
plt.plot(fpr_list,fnr_list2,label='Generic_composition')
plt.xlabel('Type I error')
plt.ylabel('Type II error')
plt.legend()
plt.show()

