
from autodp.autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism
from autodp.autodp.transformer_zoo import Composition
import matplotlib.pyplot as plt

sigma1 = 5.0
sigma2 = 8.0

gm1 = ExactGaussianMechanism(sigma1,name='GM1')
gm2 = ExactGaussianMechanism(sigma2,name='GM2')
SVT = PureDP_Mechanism(eps=0.1,name='SVT')

# run gm1 for 3 rounds
# run gm2 for 5 times
# run SVT for once

# compose them with the transformation: compose.
compose = Composition()
composed_mech = compose([gm1, gm2, SVT], [3, 5, 1])

# Query for eps given delta
delta1 = 1e-6
eps1 = composed_mech.get_approxDP(delta1)

delta2 = 1e-4
eps2 = composed_mech.get_approxDP(delta2)


# Get name of the composed object, a structured description of the mechanism generated automatically
print('Mechanism name is \"', composed_mech.name,'\"')
print('Parameters are: ',composed_mech.params)
print('epsilon(delta) = ', eps1, ', at delta = ', delta1)
print('epsilon(delta) = ', eps2, ', at delta = ', delta2)


# Get hypothesis testing interpretation so we can directly plot it
fpr_list, fnr_list = composed_mech.plot_fDP()

plt.figure(figsize = (6,6))
plt.plot(fpr_list,fnr_list)
plt.xlabel('Type I error')
plt.ylabel('Type II error')
plt.show()

