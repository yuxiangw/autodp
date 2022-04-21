
import numpy as np
import math
from sympy import *
import time

from scipy.fft import fft
from autodp import utils
import scipy.integrate as integrate

from autodp.utils import stable_logsumexp, _log1mexp

from scipy.optimize import minimize_scalar


def stable_log_diff_exp(x, y):
    # ensure that y > x
    # this function returns the stable version of log(exp(y)-exp(x)) if y > x

    s = True
    mag = y + np.log(1 - np.exp(x - y))

    return s, mag



def phi_gaussian(params,t):

    """
    The closed-form  phi-function for Gaussian mechanism.
    The log of the phi-function is (-1/2sigma^2(t^2 -it)).

    Args:
        t: the order of the  characteristic function.
        sigma: the std of the noise divide by the l2 sensitivity.

    Returns:
        The log of the phi-function evaluated at the t-th order.
    """
    sigma = params['sigma']
    result = -1.0/(2*sigma**2)*(t**2 -t*1.0j)
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





def phi_laplace_p(params, t):
    





    L = 30
    N = 5000
    b= params['b']

    dx = 2.0 * L / N
    y = np.linspace(-L, L - dx, N, dtype=np.complex128) # represent y

    """
    Return the lower and upper bound approximation
    """

    stable = (-np.abs(y)+np.abs(y-1))/b

    stable = [max(stable[max(i - 1, 0)], stable[i]) for i in range(len(stable))]
    stable = np.array(stable, dtype=np.complex128)

    stable_1 = utils.stable_logsumexp( 1.0j*stable*t -np.abs(y)/b)

    result = stable_1+np.log(dx)+np.log(1./(2.*b))

    return result


def phi_laplace_q(params, t):



    L = 60
    N = 10000
    b= params['b']

    dx = 2.0 * L / N
    y = np.linspace(-L, L - dx, N, dtype=np.complex128) # represent y

    """
    Return the lower and upper bound approximation
    """

    stable = (np.abs(y)-np.abs(y-1))/b

    stable = [max(stable[max(i - 1, 0)], stable[i]) for i in range(len(stable))]
    stable = np.array(stable, dtype=np.complex128)

    stable_1 = utils.stable_logsumexp( 1.0j*stable*t -np.abs(y-1)/b)

    result = stable_1+np.log(dx)+np.log(1./(2.*b))

    return result


def phi_laplace(params, t):
    """
    The closed-form log phi-function for Laplace mechanism.
    Args:
        t: the order of the characteristic function.
        b: the parameter for Laplace mechanism.
    Return:
        The log phi-function of Laplace mechanism.
    """
    b = params['b']

    term_1 = 1.j*t/b
    term_2 =-(1.j*t+1)/b

    result = utils.stable_logsumexp_two(utils.stable_logsumexp_two(0, np.log(1./(2*t*1.j+1)))+term_1,np.log(2*t*1.j)-
                                        np.log(2*t*1.j+1)+term_2)
    return result + np.log(0.5)










def phi_subsample_gaussian_p(params, t, phi_min = False, phi_max=False):
    """
    The phi-function of the privacy loss R.V. log(p)/log(q), i.e., phi(t) := E_p e^{t log(p)/log(q)}.
    We provide two approaches to approximate phi-function.
    In the first approach, we provide valid upper and lower bounds by setting phi_max or phi_min to be True.
    In the second approach (both phi_min = False and phi_max = False), we compute the phi-function using Gaussian
    quadrature directly.
    Args:
        phi_min: if True, the funciton provide the lower bound approximation of delta(epsilon).
        phi_max: if True, the funciton provide the upper bound approximation of delta(epsilon).
        if not phi_min and not phi_max, the function provide gaussian quadrature approximation.
        gamma: the sampling ratio.
        sigma: the std of the noise divide by the l2 sensitivity.
        In the approximation, we first truncate the 1-dim ouput space to [-L, L] and then divide it into N points.
    """

    L = 30
    N = 1000
    sigma = params['sigma']
    gamma = params['gamma']


    dx = 2.0 * L / N
    y = np.linspace(-L, L - dx, N, dtype=np.complex128) # represent y

    """
    The qua function (used for Double Quadrature method) computes e^{it log(p)/log(q)}. 
    Gaussian quadrature requires the integration interval is [-1, 1]. To integrate over [-inf, inf], we 
    first convert y (the integral in Gaussian quadrature) to new_y.
    The privacy loss R.V. log(p/q) = log(gamma * e^(2 * new_y - 1)/2 * sigma**2 + 1 -gamma)
    """
    t1 = time.time()
    def qua(y):

        new_y = y * 1.0 / (1 - y ** 2)
        stable = utils.stable_logsumexp_two(np.log(gamma)+(2*new_y-1)/(2*sigma**2),np.log(1-gamma))
        exp_term =  np.exp(1.0j * stable * t)
        density_term = utils.stable_logsumexp_two(- new_y ** 2 / (2 * sigma ** 2) + np.log(1 - gamma),- (new_y - 1) ** 2 / (2 * sigma ** 2)+ np.log(gamma) )
        inte_function = np.exp(density_term)*exp_term
        return inte_function

    # inte_f implements the integraion over an infinite intervals.
    # int_-infty^infty f(x)dx = int_-1^1 f(y/1-y^2) * (1 + y**2) / ((1 - y ** 2) ** 2).
    inte_f = lambda y: qua(y) * (1 + y ** 2) / ((1 - y ** 2) ** 2)


    if not phi_min and not phi_max:
        # Double quadrature: res computes the phi-function using Gsussian quadrature.
        res = integrate.quadrature(inte_f, -1.0, 1.0, tol=1e-15, rtol=1e-15, maxiter=200)
        result = np.log(res[0]) - np.log(np.sqrt(2 * np.pi) * sigma)
        #print('time per quad', time.time() - t1)
        return result

    """
    Return the lower and upper bound approximation
    """
    stable = utils.stable_logsumexp_two(np.log(gamma)+(2*y-1)/(2*sigma**2),np.log(1-gamma))

    if phi_min:
        # return the left riemann stable (lower bound of the privacy guarantee).
        stable= [min(stable[max(i-1,0)], stable[i]) for i in range(len(stable))]
    elif phi_max:
        stable = [max(stable[max(i - 1, 0)], stable[i]) for i in range(len(stable))]
    stable = np.array(stable, dtype=np.complex128)

    stable_1 = utils.stable_logsumexp( 1.0j*stable*t-(y-1)**2/(2*sigma**2))+np.log(gamma)-np.log(np.sqrt(2*np.pi)*sigma)
    stable_2 = utils.stable_logsumexp(1.0j*stable*t-y**2/(2*sigma**2))+np.log(1-gamma)-np.log(np.sqrt(2*np.pi)*sigma)
    p_y = utils.stable_logsumexp_two(stable_1, stable_2)
    result = p_y+np.log(dx)

    return result

def phi_subsample_gaussian_q(params, t, phi_min=False, phi_max =False):

    """
    The phi-function of the privacy loss R.V. log(q)/log(p), i.e., phi(t) := E_q e^{t log(q)/log(p)}.

    We provide two approaches to approximate the phi-function.
    In the first approach, we provide valid upper and lower bounds by setting phi_max or phi_min to be True.
    In the second approach (both phi_min = False and phi_max = False), we compute the phi-function using Gaussian
    quadrature directly. We recommend the second approach as it's more computational efficient.

    Args:
        phi_min: if True, the function provide the lower bound approximation of delta(epsilon).
        phi_max: if True, the function provide the upper bound approximation of delta(epsilon).
        if not phi_min and not phi_max, the function provide gaussian quadrature approximation.
        gamma: the sampling ratio.
        sigma: the std of the noise divide by the l2 sensitivity.
        In the approximation (first approach), we first truncate the 1-dim output space to [-L, L]
         and then divide it into N points.

    Returns:
        The phi-function evaluated at the t-th order.
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
        phi_result = -1.0*utils.stable_logsumexp_two(np.log(gamma) + (2 * new_y - 1) / (2 * sigma ** 2), np.log(1 - gamma))
        phi_result = np.exp(phi_result*1.0j*t)
        inte_function = phi_result*np.exp(-new_y**2/(2*sigma**2))
        return inte_function



    # inte_f implements the integraion over an infinite intervals.
    # int_-infty^infty f(x)dx = int_-1^1 f(y/1-y^2) * (1 + y**2) / ((1 - y ** 2) ** 2).
    inte_f = lambda y: qua(y) * (1 + y ** 2) / ((1 - y ** 2) ** 2)



    if not phi_max and not phi_min:
        # Double quadrature: res computes the phi-function using Gaussian quadrature.
        res = integrate.quadrature(inte_f, -1.0, 1.0, tol=1e-20, rtol=1e-15, maxiter=200)
        result = np.log(res[0]) - np.log(np.sqrt(2 * np.pi) * sigma)
        return result

    L = 30
    N = 1e5
    dx = 2.0 * L / N  # discretisation interval \Delta x
    y = np.linspace(-L, L-dx, N, dtype=np.complex128)
    stable = -1.0*utils.stable_logsumexp_two(np.log(gamma) + (2 * y - 1) / (2 * sigma ** 2), np.log(1 - gamma))
    if phi_min:
        # return the left riemann stable
        stable = [min(stable[max(i - 1, 0)], stable[i]) for i in range(len(stable))]
    elif phi_max:
        stable = [max(stable[max(i - 1, 0)], stable[i]) for i in range(len(stable))]
    stable = np.array(stable, dtype = np.complex128)
    exp_term = 1.0j*t*stable
    result = utils.stable_logsumexp(exp_term-y**2/(2*sigma**2))-np.log(np.sqrt(2*np.pi)*sigma)
    new_result = result+ np.log(dx)
    return new_result





