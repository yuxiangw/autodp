
import numpy as np
import math
from scipy.fft import fft
from autodp import utils
from scipy import integrate
import scipy.integrate as integrate

from autodp.utils import stable_logsumexp, stable_log_diff_exp

from scipy.optimize import minimize_scalar

def stable_logsumexp(x):
    a = np.max(x)
    return a+np.log(np.sum(np.exp(x-a)))

def _log1mexp(x):
    """  Numerically stable computation of log(1-exp(x))."""
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

def phi_gaussian(params,t):

    """
    The closed-form  phi-function for Gaussian mechanism.
    It returns log(phi(t)) = (-1/2sigma^2(t^2 -it)).
    Args:
        t: the order of the  characteristic function.
        sigma: the std of the noise divide by the l2 sensitivity.

    """
    sigma = params['sigma']
    result = -1.0/(2*sigma**2)*(t**2 -t*1.0j)
    return result

def phi_rr(params, t):
    """
    The closed-form phi-function for Randomized Response.
    Args:
        t: the order of the characteristic function.
        p: the probability in Randomized Response.
    Return:
        The log phi-function of randomized response.
    """
    p = params['p']
    term1 = np.log(p/(1.-p))
    left = np.log(p) + t*1.0j*term1
    right = np.log(1-p) + 1.0j*t*(-term1)
    a=[]
    a.append(left)
    a.append(right)
    return stable_logsumexp(a)

def phi_laplace(params, t):
    """
    The closed-form log phi-function for Laplace mechanism.
    Args:
        t: the order of the characteristic funciton.
        b: the parameter for Laplace mechanism.
    Return:
        The log phi-function of Laplace mechanism.
    """
    b = params['b']

    term_1 = 1.j*t/b
    term_2 =-(1.j*t+1)/b
    result = utils.stable_logsumexp_two(utils.stable_logsumexp_two(0, np.log(1./(2*t*1.j+1)))+term_1,np.log(2*t*1.j)-np.log(2*t*1.j+1)+term_2)
    #result =  utils.stable_logsumexp_two(np.log(1+1./(2*t*1.j+1))+term_1,np.log(1-1.0/(2*t*1.j+1))+term_2)
    result = np.log(0.5)+result
    return result








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
    N = 100000
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
    N = 100000
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
        stable = -1.0*utils.stable_logsumexp_two(np.log(gamma) + (2 * new_y - 1) / (2 * sigma ** 2), np.log(1 - gamma))
        phi_result = np.array(stable, dtype=np.complex128)
        phi_result = np.exp(phi_result*1.0j*t)
        inte_function = phi_result*np.exp(-new_y**2/(2*sigma**2))
        return inte_function

    # inte_f implements the integraion over an infinite intervals.
    # int_-infty^infty f(x)dx = int_-1^1 f(y/1-y^2) * (1 + y**2) / ((1 - y ** 2) ** 2).
    inte_f = lambda y: qua(y) * (1 + y ** 2) / ((1 - y ** 2) ** 2)



    if not phi_max and not phi_min:
        # Double quadrature: res computes the phi-function using Gsussian quadrature.
        res = integrate.quadrature(inte_f, -1.0, 1.0, tol=1e-20, rtol=1e-15, maxiter=200)
        result = np.log(res[0]) - np.log(np.sqrt(2 * np.pi) * sigma)
        return result



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





