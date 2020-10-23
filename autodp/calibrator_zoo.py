"""
'calibrator_zoo' module implements a number of ways to choose parameters of a mechanism to achieve
a pre-defined privacy guarantee.

For instance, we could calibrate noise to sensitivity in Laplace mechanism to achieve any given \eps

All calibrators inherit the autodp_core.calibrator

"""


from autodp.autodp_core import Calibrator
from scipy.optimize import minimize_scalar, root_scalar

class eps_delta_calibrator(Calibrator):
    def __init__(self):
        Calibrator.__init__(self)
        self.name = 'eps_delta_calibrator'

        # Update the function that is callable

        self.calibrate = self.param_from_eps_delta

    def param_from_eps_delta(self,mech_class,eps,delta, bounds,name=None):
        def get_eps(x,delta):
            mech = mech_class(x)
            return mech.get_approxDP(delta)

        def err(x):
            return abs(eps-get_eps(x,delta))

        results = minimize_scalar(err, method='bounded', bounds=bounds)
        if results.success and results.fun < 1e-3:
            if name:
                mech = mech_class(results.x,name=name)
            else:
                mech = mech_class(results.x)
            return mech
        else:
            print('Warning: eps_delta_calibrator fails to find a parameter')
            return None





#TODO: implement other generic and specialized privacycalibrators.
#TODO: deprecate the API from "privacy_calibrator.py"