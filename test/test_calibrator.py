from autodp.calibrator_zoo import eps_delta_calibrator
from autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism

"""
Try calibrating noise to privacy budgets.
Cases with a single parameter.

"""

calibrate = eps_delta_calibrator()
eps = 1.5
delta = 1e-6

mech1 = calibrate(ExactGaussianMechanism,eps,delta,[0,100],name='GM1')

mech2 = calibrate(PureDP_Mechanism,eps,delta,[0,100],name='Laplace')

print(mech1.name, mech1.params,mech1.get_approxDP(delta))

print(mech2.name, mech2.params, mech2.get_approxDP(delta))