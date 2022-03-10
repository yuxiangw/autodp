"""
This module contains a collections of the inverse of `privacy_calibrator'.  Given a fixed randomized algorithm and a
desired parameter `delta` it calculates the corresponding (eps,delta)-DP guarantee.

These are building blocks of many differential privacy algorithms.

In some cases, given a fixed randomized algorithm on a fixed data set, it calculates the corresponding (eps,delta)-pDP.
"""

import numpy as np
from autodp import dp_acct, rdp_acct, rdp_bank, utils
from scipy.special import comb
from scipy.stats import norm
from scipy.optimize import minimize_scalar, root_scalar


def get_eps_rdp(func, delta):
    """
    This is the generic function that uses RDP accountant and RDP function to solve for eps given delta
    :param func:
    :param delta:
    :return: The corresponding epsilon
    """
    assert(delta >= 0)
    acct = rdp_acct.anaRDPacct(m=10,m_max=10)
    acct.compose_mechanism(func)
    return acct.get_eps(delta)


def get_eps_rdp_subsampled(func, delta, prob):
    """
    This is the generic function that uses RDP accountant and RDP function to solve for eps given delta
    :param func:
    :param delta:
    :return: The corresponding epsilon
    """
    assert(delta >= 0)
    assert(prob >=0)
    if prob==0:
        return 0
    elif prob == 1:
        return get_eps_rdp(func,delta)
    else:
        acct = rdp_acct.anaRDPacct()
        acct.compose_subsampled_mechanism(func,prob)
        return acct.get_eps(delta)


# Get the eps and delta for a single Gaussian mechanism
def get_eps_gaussian(sigma, delta):
    """ This function calculates the eps for Gaussian Mech given sigma and delta"""
    assert(delta >= 0)
    func = lambda x: rdp_bank.RDP_gaussian({'sigma':sigma},x)
    return get_eps_rdp(func,delta)


def get_logdelta_ana_gaussian(sigma,eps):
    """ This function calculates the delta parameter for analytical gaussian mechanism given eps"""
    assert(eps>=0)
    s, mag = utils.stable_log_diff_exp(norm.logcdf(0.5 / sigma - eps * sigma),
                                       eps + norm.logcdf(-0.5/sigma - eps * sigma))
    return mag


def get_eps_ana_gaussian(sigma, delta):
    """ This function calculates the gaussian mechanism given sigma and delta using analytical GM"""
    # Basically inverting the above function by solving a nonlinear equation
    assert(delta >=0 and delta <=1)

    if delta == 0:
        return np.inf
    if np.log(delta) >= get_logdelta_ana_gaussian(sigma, 0.0):
        return 0.0

    def fun(x):
        if x < 0:
            return np.inf
        else:
            return get_logdelta_ana_gaussian(sigma, x) - np.log(delta)

    eps_upperbound = 1/2/sigma**2+1/sigma*np.sqrt(2*np.log(1/delta))
    results = root_scalar(fun,bracket=[0, eps_upperbound])
    if results.converged:
        return results.root
    else:
        raise RuntimeError(f"Failed to find epsilon: {results.flag}")

def eps_generalized_gaussian(x, sigma, delta,k, c, c_tilde):
    """
    submodule for generalized SVT with Gaussian noise
    we want to partition c into [c/c'] parts, each part using (k choose c')
    need to check whether (k choose c') > log(1/delta')
    k is the maximam number of queries to answer for each chunk
    x is log delta for each chunk, it needs to be negative
    :param x:
    :param sigma:
    :param delta:
    :return:
    """
    acct = dp_acct.DP_acct()
    per_delta = np.exp(x) # per_delta for each c' chunk
    coeff = comb(k,c_tilde)
    assert per_delta < 1.0/(coeff)
    #compute the eps per step with 1/(sigma_1**2) + sqrt(2/simga_1**2 *(log k + log(1/epr_delta)))
    # compose eps for each chunk
    while c:
        if c>= c_tilde:
            c = c - c_tilde
        else:
            # the remaining part c // c_tilde
            c = 0
            c_tilde = c
            coeff = comb(k, c_tilde)
        per_eps = (1.0+c_tilde) / (2*sigma ** 2) + np.sqrt((1.0 +c_tilde) / (2*sigma ** 2) * (np.log(coeff) - x))
        acct.update_DPlosses(per_eps, per_delta)

    compose_eps = acct.get_eps(delta)
    return compose_eps


def get_eps_laplace(b,delta):
    assert(delta >= 0)
    func = lambda x: rdp_bank.RDP_laplace({'b':b},x)
    return get_eps_rdp(func,delta)


def get_eps_randresp(p,delta):
    assert(delta >= 0)
    func = lambda x: rdp_bank.RDP_randresponse({'p':p},x)
    return get_eps_rdp(func, delta)



def get_eps_randresp_optimal(p,delta):
    assert(delta >= 0)
    if p<0.5:
        p = 1-p

    if p==0 or p==1:
        return np.inf

    eps_max = np.log(p) - np.log(1-p)
    if delta == 0:
        return eps_max
    elif delta >= 2*p - 1:
        return 0.0
    else:
        return np.log(p-delta) - np.log(1 - p)
