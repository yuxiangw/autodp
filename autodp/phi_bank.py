# This module implements the log of characteristic  function templates for various
# mechanisms (see  https://arxiv.org/abs/2106.08567)
# We will use a pair of characteristic function to describe each mechanism.
# The conversion back to delta(eps) or eps(delta) will use numerical inversion, which details in converter.py


import numpy as np
from autodp import utils
import scipy.integrate as integrate
from autodp.utils import stable_logsumexp, _log1mexp






def phi_gaussian(params, t):
    """
    The closed-form log phi-function  of Gaussian mechanism.
    The log of the phi-function is (-1/2sigma^2(t^2 -it)).

    Args:
        t: the order of the  characteristic function.
        sigma: the std of the noise divide by the l2 sensitivity.

    Returns:
        The log of the phi-function evaluated at the t-th order.
    """
    sigma = params['sigma']
    result = -1.0 / (2 * sigma ** 2) * (t ** 2 - t * 1.0j)
    return result


def phi_rr_p(params, t):
    """
    The closed-form log phi-function  for Randomized Response, where
    the privacy loss R.V. is drawn from the distribution P.

    Args:
        t: the order of the characteristic function.
        params: contains two parameters: p and q.
        p: the probability to output one for dataset X.
        q: the probability to output one for dataset X'. default q = 1 - p.

    Return:
        The log phi-function of randomized response.
        """
    p = params['p']
    # generalized randomized response
    q = params['q']
    term_1 = np.log(p / q)
    term_2 = np.log((1 - p) / (1 - q))
    a = []
    left = np.log(p) + t * 1.0j * term_1
    right = np.log(1 - p) + 1.0j * t * term_2
    a.append(left)
    a.append(right)
    return stable_logsumexp(a)


def phi_rr_q(params, t):
    """
    The closed-form log phi-function for Randomized Response, where
    the privacy loss R.V. is drawn from the distribution Q
    (see Definition 12 in https://arxiv.org/pdf/2106.08567.pdf)

    The generalized randomize response can represent any pure-DP mechanisms.
    Args:
        t: the order of the characteristic function.
        params: contains two parameters: p and q.
        p: the probability to output one for dataset X.
        q: the probability to output one for dataset X'. default q = 1 - p.

    Return:
        The log phi-function of randomized response.
    """
    p = params['q']
    q = params['p']
    term_1 = np.log(p / q)
    term_2 = np.log((1 - p) / (1 - q))
    a = []
    left = np.log(p) + t * 1.0j * term_1
    right = np.log(1 - p) + 1.0j * t * term_2
    a.append(left)
    a.append(right)
    return stable_logsumexp(a)


def phi_laplace(params, t):
    """
    The closed-form log phi-function fof  Laplace mechanism.
    Args:
        t: the order of the characteristic function.
        b: the parameter for Laplace mechanism.

    Return:
        The log phi-function of Laplace mechanism.
    """
    b = params['b']

    term_1 = 1.j * t / b
    term_2 = -(1.j * t + 1) / b

    result = utils.stable_logsumexp_two(utils.stable_logsumexp_two(0, np.log(1. / (2 * t * 1.j + 1))) + term_1,
                                        np.log(2 * t * 1.j) -
                                        np.log(2 * t * 1.j + 1) + term_2)
    return result + np.log(0.5)


def phi_subsample_gaussian_p(params, t, remove_only=False):
    """
    Return the  log phi-function of the poisson subsample Gaussian mechanism (evaluated at the t-th order).
    phi(t) := E_p e^{t log(p)/log(q)}.
    For Poisson sampling, if  P: N(1, sigma^2) and Q:(0, sigma^2) dominate the base mechanism for both adding
     and removing neighboring, relationship, then after sampling with probability 'prob'
    new_p: (1-prob)*Q + prob*P; new_q: Q will dominates the sampled mechanism for remove only relationship.
    new_p: P; new_q: (1-prob)P + prob*Q will dominates the sampled mechanism for add only relationship.
    For details, please refer to Theorem 11 in https://arxiv.org/pdf/2106.08567.pdf

    The phi-function of the subsample Gaussian is not symmetric. This function implements the log phi-function
    phi(t) := E_p e^{t log(p)/log(q)} using Gaussian quadrature directly.
     Args:
        gamma: the sampling ratio.
        sigma: the std of the noise divide by the l2 sensitivity.
    """

    sigma = params['sigma']
    gamma = params['gamma']

    """
    The qua function (used for Double Quadrature method) computes e^{it log(p)/log(q)} 
    Gaussian quadrature requires the integration interval is [-1, 1]. To integrate over [-inf, inf], we 
    first convert y (the integral in Gaussian quadrature) to new_y.
    The privacy loss R.V. log(p/q) = log(gamma * e^(2 * new_y - 1)/2 * sigma**2 + 1 -gamma)
    """
    def qua(y):
        new_y = y * 1.0 / (1 - y ** 2)
        if remove_only is True:
            # stable computes log(new_p(new_y)/new_q(new_y))
            stable = utils.stable_logsumexp_two(np.log(gamma) + (2 * new_y-1) / (2 * sigma ** 2), np.log(1 - gamma))
            exp_term = np.exp(1.0j * stable * t)
            # density_term computes the pdf of new_p: (1-prob)*Q + prob*P
            density_term = utils.stable_logsumexp_two(- (new_y-1) ** 2 / (2 * sigma ** 2) + np.log(gamma), - (new_y)
                                                                ** 2 / (2 * sigma ** 2) + np.log(1-gamma))
            inte_function = np.exp(density_term) * exp_term


        else:
            # returns the add_only result.
            # new p(x): 1/sqrt{2pi*sigma**2}e^{-(x-1)^2/2sigma**2} new q(x) = (1-gamma)/(sqrt{2pi*sigma^2}*e^{-(x-1)**2
            # /2sigma**2} + gamma/sqrt{2pi*sigma**2}*e^{-(x)**2/2sigma**2}
            stable = -utils.stable_logsumexp_two(np.log(1-gamma), np.log(gamma)+(1-2*new_y)/(2*sigma**2))
            # Stable is for log new_P(x)/new_Q(x)
            exp_term = np.exp(1.0j * stable * t)
            density_term = -(new_y-1) ** 2 / (2 * sigma ** 2)
            inte_function = np.exp(density_term) * exp_term
        return inte_function

    # inte_f implements the integration over an infinite intervals.
    # int_-infty^infty f(x)dx = int_-1^1 f(y/(1-y^2)) * (1 + y**2) / ((1 - y ** 2) ** 2).
    inte_f = lambda y: qua(y) * (1 + y ** 2) / ((1 - y ** 2) ** 2)

    # Double quadrature: res computes the phi-function using Gaussian quadrature.
    res = integrate.quadrature(inte_f, -1.0, 1.0, tol=1e-15, rtol=1e-15, maxiter=100)
    result = np.log(res[0]) - np.log(np.sqrt(2 * np.pi) * sigma)
    return result


def phi_subsample_gaussian_q(params, t, remove_only=False):
    """
    Return the  log phi-function of the poisson subsample Gaussian mechanism (evaluated at the t-th order).
    phi(t) := E_q e^{t log(q)/log(p)}.

    For Poisson sampling, if  P: N(1, sigma^2) and Q:(0, sigma^2) dominate the base mechanism for both adding
     and removing neighboring, relationship, then after sampling with probability 'prob'
    new_p: (1-prob)*Q + prob*P; new_q: Q will dominates the sampled mechanism for remove only relationship.
    new_p: P; new_q: (1-prob)P + prob*Q will dominates the sampled mechanism for add only relationship.
    For details, please refer to Theorem 11 in https://arxiv.org/pdf/2106.08567.pdf

    Args:
        gamma: the sampling ratio.
        sigma: the std of the noise divide by the l2 sensitivity.

    Returns:
        The log phi-function evaluated at the t-th order.
    """

    sigma = params['sigma']
    gamma = params['gamma']

    """
    The qua function (used for Double Quadrature method) computes e^{it log(p)/log(q)}. 
    Gaussian quadrature requires the integration interval is [-1, 1]. To integrate over [-inf, inf], we 
    first convert y (the integral in Gaussian quadrature) to new_y.
    The privacy loss R.V. log(p/q) = -log(gamma * e^(2 * new_y - 1)/2 * sigma**2 + 1 -gamma)
    """

    def qua(y):
        new_y = y * 1.0 / (1 - y ** 2)
        if remove_only is True:

            log_q_p = -1.0 * utils.stable_logsumexp_two(np.log(gamma) + (2 * new_y - 1) / (2 * sigma ** 2),
                                                           np.log(1 - gamma))
            phi_result = np.exp(log_q_p * 1.0j * t)
            inte_function = phi_result * np.exp(-new_y ** 2 / (2 * sigma ** 2))
        else:
            # for add_only: log_q_p is log(new_Q(new_y)/new_P(new_y))
            log_q_p = utils.stable_logsumexp_two(np.log(1 - gamma), np.log(gamma) + (1 - 2 * new_y) / (2 * sigma ** 2))

            exp_term = np.exp(log_q_p * 1.0j *t)
            # density is (1-gamma)*P + gamma*Q, which is (1-gamma)*e^{-(x-1)^2/2sigma^2} + gamma * np.exp(-x^2/2sigma^2)
            density_term = utils.stable_logsumexp_two(-(new_y-1)**2/(2*sigma**2)+np.log(1-gamma),np.log(gamma)-(new_y)
                                                      **2/(2*sigma**2))
            inte_function = exp_term * np.exp(density_term)

        return inte_function

    # inte_f implements the integration over an infinite intervals.
    # int_-infty^infty f(x)dx = int_-1^1 f(y/1-y^2) * (1 + y**2) / ((1 - y ** 2) ** 2).
    inte_f = lambda y: qua(y) * (1 + y ** 2) / ((1 - y ** 2) ** 2)


    # Double quadrature: res computes the phi-function using Gaussian quadrature.
    res = integrate.quadrature(inte_f, -1.0, 1.0, tol=1e-15, rtol=1e-15, maxiter=100)
    result = np.log(res[0]) - np.log(np.sqrt(2 * np.pi) * sigma)
    return result


