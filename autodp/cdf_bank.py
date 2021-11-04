
import numpy as np
import math
import pickle
from scipy import integrate
from autodp import utils,rdp_bank,phi_bank
from scipy import special
from scipy.stats import norm
import scipy.integrate as integrate

from scipy.optimize import minimize_scalar


def _log1mexp(x):
    """ from pate Numerically stable computation of log(1-exp(x))."""
    if x < -1:
        return math.log1p(-math.exp(x))
    elif x < 0:
        return math.log(-math.expm1(x))
    elif x == 0:
        return -np.inf
    else:
        raise ValueError("Argument must be non-positive.")


def stable_log_diff_exp(x):
    # ensure that y > x
    # this function returns the stable version of log(exp(y)-exp(x)) if y > x

    mag = np.log(1 - np.exp(x - 0))

    return mag



def cdf_approx(log_phi, ell, tbd=None):
    """
     This function computes the CDF of privacy loss R.V. using Gaussian quadrature.

     Args:
        log_phi: the log of characteristic (phi)  function.
        ell: return the CDF when the privacy loss RV is evaluated at ell-th order
        tbd: undermined parameter

    :return: CDF
    """

    def qua(t):
        """
        This part first projects the infinite intergral to [-1,1]
        """
        new_t = t*1.0/(1-t**2)
        if tbd is not None:
            phi_result = [log_phi(x, tbd) for x in new_t]
        else:
            phi_result = [log_phi(x) for x in new_t]
        phi_result = np.array(phi_result, dtype=np.complex128)
        inte_function = 1.j/new_t * np.exp(-1.j*new_t*ell)*np.exp(phi_result)
        return inte_function

    inte_f = lambda t: qua(t)*(1+t**2)/((1-t**2)**2)
    #print('h')
    # n is the maximum sampling point, usually 700 is enough.
    res = integrate.fixed_quad(inte_f, -1.0, 1.0, n =700)
    #res = integrate.quadrature(inte_f, -1.0, 1.0, maxiter=200)
    result = res[0]
    error = res[1]
    #print('quadrature result', np.real(result)/(2*np.pi)+0.5, 'error range', error)
    return result / (2 * np.pi) + 0.5
    return np.real(result)/(2*np.pi)+0.5

    """
    # the following approach uses Rieman Sum
    L = 100
    N = 10**5
    dx = 2.0 * L / N  # discretisation interval \Delta x
    t = np.linspace(-L, L - dx, N, dtype=np.complex128)
    t[int(N / 2)] = t[int(N / 2) + 1]
    phi_result = [log_phi(each_t) for each_t in t]
    phi_result = np.array(phi_result, dtype=np.complex128)
    term_1 = np.log(1.0j / t) - 1.j * t * ell + phi_result
    log_result = utils.stable_logsumexp(term_1)
    riemann_result = np.real(np.exp(log_result) * dx / (2 * np.pi) + 0.5)
    print('ell', ell, 'riemann cdf', riemann_result, min(max(riemann_result, 0), 1))
    return  min(max(riemann_result,0),1)
    """

