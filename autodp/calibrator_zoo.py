"""
'calibrator_zoo' module implements a number of ways to choose parameters of a mechanism to achieve
a pre-defined privacy guarantee.

For instance, we could calibrate noise to sensitivity in Laplace mechanism to achieve any given \eps

All calibrators inherit the autodp_core.calibrator

"""

from math import exp, sqrt
from scipy.special import erf
from autodp.autodp_core import Calibrator
from scipy.optimize import minimize_scalar, root_scalar

class eps_delta_calibrator(Calibrator):
    """
    Noting there exists multiple parameters in mechanisms. we must specify which parameter requires optimization.
    """
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
            raise RuntimeError(f"eps_delta_calibrator fails to find a parameter: {results.message}")


class generalized_eps_delta_calibrator(Calibrator):
    """
    Noting there exists multiple parameters in mechanisms. we must specify which parameter requires optimization.
    """
    def __init__(self):
        Calibrator.__init__(self)
        self.name = 'generalized_eps_delta_calibrator'

        # Update the function that is callable

        self.calibrate = self.param_from_eps_delta

    def param_from_eps_delta(self,mech_class,eps,delta, bounds, params=None, para_name=None, name=None):
        """
        params defines all parameters, include those need to optimize over
        para_name is the parameter that we need to tune

        for example, in subsampleGaussian mechanism
        params = {'prob':prob, 'sigma':sigma, 'coeff':coeff}, where coeff and sigma are given as the fixed input
        """
        def get_eps(x,delta):
            if params is None:
                # params is None implies the mechanism only has one parameter
                mech = mech_class(x)
            else:
                params[para_name] = x
                mech = mech_class(params)
            return mech.get_approxDP(delta)

        def err(x):
            return abs(eps-get_eps(x,delta))

        results = minimize_scalar(err, method='bounded', bounds=bounds)
        if results.success and results.fun < 1e-3:

            if params is None:
                mech = mech_class(results.x, name=name)
            else:
                params[para_name] = results.x
                mech = mech_class(params,name=name)
            return mech
        else:
            raise RuntimeError(f"eps_delta_calibrator fails to find a parameter: {results.message}")



class ana_gaussian_calibrator(Calibrator):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

    Modified from https://github.com/BorjaBalle/analytic-gaussian-mechanism/blob/master/agm-example.py

    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    tol : error tolerance for binary search (tol > 0)
    Output:
    params : a dictionary that contains field `sigma' --- the standard deviation of Gaussian noise needed to achieve
        (epsilon,delta)-DP under global sensitivity 1

    Note that this one does not support composition or subsample
    """


    def __init__(self):
        Calibrator.__init__(self)
        self.name = 'ana_gaussian_calibrator'

        # Update the function that is callable

        self.calibrate = self.param_from_eps_delta

    def param_from_eps_delta(self,mech_class,eps,delta,name=None, tol=1.e-12):
        def Phi(t):
            return 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))

        def caseA(epsilon, s):
            return Phi(sqrt(epsilon * s)) - exp(epsilon) * Phi(-sqrt(epsilon * (s + 2.0)))

        def caseB(epsilon, s):
            return Phi(-sqrt(epsilon * s)) - exp(epsilon) * Phi(-sqrt(epsilon * (s + 2.0)))

        def doubling_trick(predicate_stop, s_inf, s_sup):
            while (not predicate_stop(s_sup)):
                s_inf = s_sup
                s_sup = 2.0 * s_inf
            return s_inf, s_sup

        def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
            s_mid = s_inf + (s_sup - s_inf) / 2.0
            while (not predicate_stop(s_mid)):
                if (predicate_left(s_mid)):
                    s_sup = s_mid
                else:
                    s_inf = s_mid
                s_mid = s_inf + (s_sup - s_inf) / 2.0
            return s_mid

        delta_thr = caseA(eps, 0.0)

        if (delta == delta_thr):
            alpha = 1.0
            sigma = alpha / sqrt(2.0 * eps)

        else:
            if (delta > delta_thr):
                predicate_stop_DT = lambda s: caseA(eps, s) >= delta
                function_s_to_delta = lambda s: caseA(eps, s)
                predicate_left_BS = lambda s: function_s_to_delta(s) > delta
                function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)

            else:
                predicate_stop_DT = lambda s: caseB(eps, s) <= delta
                function_s_to_delta = lambda s: caseB(eps, s)
                predicate_left_BS = lambda s: function_s_to_delta(s) < delta
                function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

            predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - delta) <= tol

            s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
            s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
            alpha = function_s_to_alpha(s_final)
            sigma = alpha / sqrt(2.0 * eps)


        if name:
            mech = mech_class(sigma,name=name)
        else:
            mech = mech_class(sigma)
        return mech



#TODO: implement other generic and specialized privacycalibrators.
#TODO: deprecate the API from "privacy_calibrator.py"

