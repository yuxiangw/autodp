
from autodp.autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism
from autodp.autodp.transformer_zoo import Composition, AmplificationBySampling
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

poisson_sample = AmplificationBySampling(PoissonSampling=True)
subsample = AmplificationBySampling(PoissonSampling=False)

prob = 0.1
coeffs = [30,50,10]

composed_mech = compose([gm1, gm2, SVT], coeffs)

composed_poissonsampled_mech = compose([poisson_sample(gm1,prob),
                                        poisson_sample(gm2,prob),
                                        poisson_sample(SVT,prob)],
                                       coeffs)

composed_poissonsampled_mech1 = compose([poisson_sample(gm1,prob,improved_bound_flag=True),
                                        poisson_sample(gm2,prob,improved_bound_flag=True),
                                        poisson_sample(SVT,prob,improved_bound_flag=True)],
                                       coeffs)



# Now let's do subsampling. First we need to use replace-one version of the base mechanisms.
gm1.replace_one = True
gm2.replace_one = True
SVT.replace_one = True

composed_subsampled_mech = compose([subsample(gm1,prob),
                                    subsample(gm2,prob),
                                    subsample(SVT,prob)],
                                   coeffs)

composed_subsampled_mech1 = compose([subsample(gm1,prob,improved_bound_flag=True),
                                    subsample(gm2,prob,improved_bound_flag=True),
                                    subsample(SVT,prob,improved_bound_flag=True)],
                                   coeffs)

# Query for eps given delta
delta1 = 1e-6
eps1 = composed_mech.get_approxDP(delta1)

delta2 = 1e-4
eps2 = composed_mech.get_approxDP(delta2)


# Get name of the composed object, a structured description of the mechanism generated automatically
print('---------------------------------------------------')
print('Mechanism name is \"', composed_mech.name,'\"')
print('Parameters are: ',composed_mech.params)
print('epsilon(delta) = ', eps1, ', at delta = ', delta1)
print('epsilon(delta) = ', eps2, ', at delta = ', delta2)


eps1a = composed_poissonsampled_mech.get_approxDP(delta1)
eps2a = composed_poissonsampled_mech.get_approxDP(delta2)

eps1aa = composed_poissonsampled_mech1.get_approxDP(delta1)
eps2aa = composed_poissonsampled_mech1.get_approxDP(delta2)

# Get name of the composed object, a structured description of the mechanism generated automatically
print('---------------------------------------------------')
print('Mechanism name is \"', composed_poissonsampled_mech.name,'\"')
print('Parameters are: ',composed_poissonsampled_mech.params)
print('epsilon(delta) = ', eps1a, ', at delta = ', delta1)
print('epsilon(delta) = ', eps2a, ', at delta = ', delta2)
print('------- If qualified for the improved bounds --------')
print('epsilon(delta) = ', eps1aa, ', at delta = ', delta1)
print('epsilon(delta) = ', eps2aa, ', at delta = ', delta2)


eps1b = composed_subsampled_mech.get_approxDP(delta1)
eps2b = composed_subsampled_mech.get_approxDP(delta2)

eps1bb = composed_subsampled_mech1.get_approxDP(delta1)
eps2bb = composed_subsampled_mech1.get_approxDP(delta2)

# Get name of the composed object, a structured description of the mechanism generated automatically
print('---------------------------------------------------')
print('Mechanism name is \"', composed_subsampled_mech.name,'\"')
print('Parameters are: ',composed_subsampled_mech.params)
print('epsilon(delta) = ', eps1b, ', at delta = ', delta1)
print('epsilon(delta) = ', eps2b, ', at delta = ', delta2)
print('------- If qualified for the improved bounds --------')
print('epsilon(delta) = ', eps1bb, ', at delta = ', delta1)
print('epsilon(delta) = ', eps2bb, ', at delta = ', delta2)

# # Get hypothesis testing interpretation so we can directly plot it
# fpr_list, fnr_list = composed_mech.plot_fDP()
#
# plt.figure(figsize = (6,6))
# plt.plot(fpr_list,fnr_list)
# plt.xlabel('Type I error')
# plt.ylabel('Type II error')
# plt.show()

