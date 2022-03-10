import sys
import os
from autodp.calibrator_zoo import eps_delta_calibrator,generalized_eps_delta_calibrator, ana_gaussian_calibrator
from autodp import rdp_bank
from autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism,SubsampleGaussianMechanism, GaussianMechanism, ComposedGaussianMechanism, LaplaceMechanism
from autodp.transformer_zoo import Composition, AmplificationBySampling

"""
Try calibrating noise to privacy budgets.
Cases 1: single parameter, no subsample or composition

"""

calibrate = eps_delta_calibrator()
ana_calibrate = ana_gaussian_calibrator()
eps = 0.1
delta = 1e-6

mech1 = calibrate(ExactGaussianMechanism,eps,delta,[0,100],name='GM')
mech2 = ana_calibrate(ExactGaussianMechanism, eps, delta, name='Ana_GM')
print(mech1.name, mech1.params, mech1.get_approxDP(delta))
print(mech2.name, mech2.params, mech2.get_approxDP(delta))


"""

Cases 2: Test calibration with Gaussian under composition, coeff is the number of composition
We now have multiple parameters --- params['coeff'] and params['sigma']. 
The coeff is fixed and the calibrator optimize over sigma. We use para_name to denote the parameter that we want to optimize.

"""
coeff = 20
general_calibrate = generalized_eps_delta_calibrator()
params = {}
params['sigma'] = None
params['coeff'] = 20

mech3 = general_calibrate(ComposedGaussianMechanism, eps, delta, [0,1000],params=params,para_name='sigma', name='Composed_Gaussian')
print(mech3.name, mech3.params, mech3.get_approxDP(delta))

"""
Cases 3: Test calibration with SubsampledGaussian 
We now have three parameters --- params['coeff'], params['prob'] and params['sigma']. 
The coeff and prob are fixed and the calibrator optimize over sigma. We use para_name to denote the parameter that we want to optimize.

"""
params['prob'] = 0.01
mech4 = general_calibrate(SubsampleGaussianMechanism, eps, delta, [0,1000],params=params,para_name='sigma', name='Subsampled_Gaussian')
print(mech4.name, mech4.params, mech4.get_approxDP(delta))


"""

Cases 4: single parameter for Laplace mechanism, no subsample or composition

"""

calibrate = generalized_eps_delta_calibrator()

eps = 0.1
delta = 1e-6
mech = calibrate(LaplaceMechanism,eps,delta,[0,100],name='Laplace')
print(mech.name, mech.params, mech.get_approxDP(delta))


"""
Case 5:  Calibration of complexed mechanism coming out of composition / sampling
"""


from autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism
from autodp.transformer_zoo import Composition

# We will consider the following complex mechanism fronm "example_composition.py"

# run gm1 for 3 rounds
# run gm2 for 5 times
# run SVT for once
# compose them with the transformation: compose.

# What we will have to do is to define a new mechanism class based on the above object,
# which involves two modular steps

# The first step is to package this in a function where the input are the parameters,
# and the output is the mechanism object

def create_complex_mech(sigma1,sigma2,eps, coeffs):
    gm1 = ExactGaussianMechanism(sigma1, name='GM1')
    gm2 = ExactGaussianMechanism(sigma2, name='GM2')
    SVT = PureDP_Mechanism(eps=eps, name='SVT')

    # run gm1 for 3 rounds
    # run gm2 for 5 times
    # run SVT for once
    # compose them with the transformation: compose.
    compose = Composition()
    composed_mech = compose([gm1, gm2, SVT], coeffs)
    return composed_mech

# next we can create it as a mechanism class, which requires us to inherit the base mechanism class,
#  which we import now

from autodp.autodp_core import Mechanism

class Complex_Mechanism(Mechanism):
    def __init__(self, params, name="A_Good_Name"):
        self.name = name
        self.params = params
        mech = create_complex_mech(params['sigma1'], params['sigma2'], params['eps'], params['coeffs'])
        # The following will set the function representation of the complex mechanism
        # to be the same as that of the mech
        self.set_all_representation(mech)


# Now one can calibrate the mechanism to achieve a pre-defined privacy budget

# Let's say we want to fix sigma1 and sigma2 while tuning the eps parameter from SVT

sigma1 = 5.0
sigma2 = 8.0

coeffs = [3, 5, 1]

# Let's say we are to calibrate the noise to the following privacy budget
eps_budget = 2.5

delta_budget= 1e-6

# declare a general_calibrate "Calibrator"

general_calibrate = generalized_eps_delta_calibrator()
params = {}
params['sigma1'] = sigma1
params['sigma2'] = sigma2
params['coeffs'] = coeffs
params['eps'] = None


mech_bob = general_calibrate(Complex_Mechanism, eps_budget, delta_budget, [0,50],params=params,
                             para_name='eps',
                             name='Complex_Mech_Bob')

print(mech_bob.name, mech_bob.params, mech_bob.get_approxDP(delta))


