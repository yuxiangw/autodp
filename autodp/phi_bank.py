
import numpy as np
import math
from scipy.fft import fft
from autodp import utils,rdp_bank
from scipy import integrate
import scipy.integrate as integrate
#contains various characteristic functions
#for each function, it returns the log(phi(t))


from scipy.optimize import minimize_scalar

def stable_logsumexp(x):
    a = np.max(x)
    return a+np.log(np.sum(np.exp(x-a)))

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

def phi_gaussian(params,t,log_off=False, coeff= 100):

    """
    t: the parameter for characteristic function
    the closed-form expression of Gaussian is -1/2sigma^2(t^2 -it)
    """
    assert not log_off
    sigma = params['sigma']
    result = -1.0/(2*sigma**2)*(t**2 -t*1.0j)*coeff
    #print('t',t, 'log phi(t)',result)
    return result

def phi_rr(params, t, coeff=1):
    p = params['p']
    term1 = np.log(p/(1.-p))
    left = np.log(p) + t*1.0j*term1
    right = np.log(1-p) + 1.0j*t*(-term1)
    a=[]
    a.append(left)
    a.append(right)
    return stable_logsumexp(a)*coeff

def phi_laplace(params,t,coeff= 1):
    """
    b the parameter for laplace
    """
    b = params['b']

    term_1 = 1.j*t/b
    term_2 =-(1.j*t+1)/b
    result = utils.stable_logsumexp_two(utils.stable_logsumexp_two(0, np.log(1./(2*t*1.j+1)))+term_1,np.log(2*t*1.j)-np.log(2*t*1.j+1)+term_2)
    #result =  utils.stable_logsumexp_two(np.log(1+1./(2*t*1.j+1))+term_1,np.log(1-1.0/(2*t*1.j+1))+term_2)
    result = np.log(0.5)+result
    #print('result of phi', result)
    return result*coeff
def phi_gaussian_ana(params, t, coeff= 100):
    """
    assume the input is already discrete
    [-T, T] divide by N
    use the polar form to adds them up
    """
    #when sigma is large, we shall set L to be larger, since the variance of the privacy loss RV is now very small
    # in first step, we splits the possible output space of the privacy loss RV within [-T, T]
    L = 1000
    N = 200000
    sigma = params['sigma']
    dx = 2.0 * L / N  # discretisation interval \Delta x
    y = np.linspace(-L, L - dx, N, dtype=np.complex128)
    l = (2*y-1.)/(2*sigma**2)
    p_y = 1 / (np.sqrt(np.pi * 2) * sigma) * np.exp((y - 1) ** 2 / (-2 * sigma ** 2))
    exp_term = np.exp(1.0j * t * l)
    result = np.sum([exp_term * p_y]) * dx   #

    return np.log(result)*coeff


def phi_gaussian_fft(params,  log_off=False):
    """
    use the fft
    return t must be integer
    """

    L = 20 * np.pi
    N = 20000000
    N_phi = int(2*np.pi*N/L)
    sigma = params['sigma']
    dx = 2.0 * L / N  # discretisation interval \Delta x
    x = np.linspace(-L, L - dx, N, dtype=np.complex128)  # grid for the numerical integration
    ## use y = (2sigma^2l +1)/2
    x = (2 * sigma ** 2 * x + 1) / 2.
    x = 1.0 / (np.sqrt(np.pi * 2) * sigma) * np.exp((x - 1) ** 2 / -(2 * sigma ** 2))
    fft_x = fft(x).conj()
    t = np.linspace(-N_phi / 2, N_phi/ 2 - 1, N_phi, dtype=np.complex128)
    coeff = dx * np.exp(1.0j * t[int(N_phi / 2):] * (-L))
    coeff_ori = dx * np.exp(1.0j * t[:int(N_phi/2)] * (-L))
    result = []
    for i in range(int(N_phi / 2)):
        result.append(coeff_ori[i] * fft_x[int(N * dx * t[i] / (2 * np.pi))] * sigma ** 2)
    for i in range(int(N_phi / 2), N_phi):
        result.append(coeff[i - int(N_phi / 2)] * fft_x[int(N * dx * t[i] / (2 * np.pi))] * sigma ** 2)
    # result has shape of N from[-N/2, N/2-1]

    return np.log(result)
    #result = coeff * fft_x[idx] * sigma ** 2
    #return np.log(result)




def phi_subsample_gaussian_p(params, t, coeff=1000, phi_min = False, phi_max=False):
    """
    consider the phi funciton  E_p log(p)/log(q)
    if phi_min = True, provide the lower bound approximation of delta(epsilon)
    if phi_max = True, provide the upper bound approximation of delta(epsilon)
    if not phi_min and not phi_max, provide gaussian quadrature approximation
    gamma: sampling ratio
    sigma: noise scale

    """

    # truncate the output range to [-L, L] and divide it into 2N parts.
    L = 30
    N = 100000
    sigma = params['sigma']
    gamma = params['gamma']

    dx = 2.0 * L / N  # discretization interval \Delta x
    y = np.linspace(-L, L - dx, N, dtype=np.complex128) # represent y

    """
    qua is the phi function, we will use qua in Gaussian quadrature
    """
    def qua(y):
        # print('integrade t', len(t)) #max iter indicates how many sampled points
        new_y = y * 1.0 / (1 - y ** 2)
        stable = utils.stable_logsumexp_two(np.log(gamma)+(2*new_y-1)/(2*sigma**2),np.log(1-gamma))
        exp_term =  np.exp(1.0j * stable * t)
        density_term = utils.stable_logsumexp_two(- new_y ** 2 / (2 * sigma ** 2) + np.log(1 - gamma),- (new_y - 1) ** 2 / (2 * sigma ** 2)+ np.log(gamma) )
        inte_function = np.exp(density_term)*exp_term
        return inte_function

    inte_f = lambda y: qua(y) * (1 + y ** 2) / ((1 - y ** 2) ** 2)

    res = integrate.quadrature(inte_f, -1.0, 1.0,tol=1e-15,rtol=1e-15, maxiter=200)
    result = np.log(res[0])-np.log(np.sqrt(2*np.pi)*sigma)
    #print('phi for p', result*coeff, 'error', res[1])
    if not phi_min and not phi_max:
        return result*coeff

    """
    Return the lower and upper bound approximation
    """
    stable = utils.stable_logsumexp_two(np.log(gamma)+(2*y-1)/(2*sigma**2),np.log(1-gamma))

    if phi_min:
        # return the left riemann stable
        stable= [min(stable[max(i-1,0)], stable[i]) for i in range(len(stable))]
    elif phi_max:
        stable = [max(stable[max(i - 1, 0)], stable[i]) for i in range(len(stable))]
    stable = np.array(stable, dtype=np.complex128)

    stable_1 = utils.stable_logsumexp( 1.0j*stable*t-(y-1)**2/(2*sigma**2))+np.log(gamma)-np.log(np.sqrt(2*np.pi)*sigma)
    stable_2 = utils.stable_logsumexp(1.0j*stable*t-y**2/(2*sigma**2))+np.log(1-gamma)-np.log(np.sqrt(2*np.pi)*sigma)
    p_y = utils.stable_logsumexp_two(stable_1, stable_2)
    result = p_y+np.log(dx)

    return result* coeff

def phi_subsample_gaussian_q(params, t, coeff = 1000, phi_min=False, phi_max =False):

    """
       consider the phi funciton  E_q log(q)/log(p)
       if phi_min = True, provide the lower bound approximation of delta(epsilon)
       if phi_max = True, provide the upper bound approximation of delta(epsilon)
       if not phi_min and not phi_max, provide gaussian quadrature approximation
       gamma: sampling ratio
       sigma: noise scale
       coeff: the number of composition

    """

    # in first step, we splits the possbile output space of the privacy loss RV within [-T, T]
    L = 30
    N = 100000
    sigma = params['sigma']
    gamma = params['gamma']
    #the following uses quadrature method

    def qua(y):
        # print('integrade t', len(t)) #max iter indicates how many sampled points
        new_y = y * 1.0 / (1 - y ** 2)
        stable = -1.0*utils.stable_logsumexp_two(np.log(gamma) + (2 * new_y - 1) / (2 * sigma ** 2), np.log(1 - gamma))
        phi_result = np.array(stable, dtype=np.complex128)
        phi_result = np.exp(phi_result*1.0j*t)
        inte_function = phi_result*np.exp(-new_y**2/(2*sigma**2))
        return inte_function

    inte_f = lambda y: qua(y) * (1 + y ** 2) / ((1 - y ** 2) ** 2)
    from scipy import integrate
    res = integrate.quadrature(inte_f, -1.0, 1.0,tol=1e-20,rtol=1e-15, maxiter=200)
    result = np.log(res[0])-np.log(np.sqrt(2*np.pi)*sigma)
    #return Gaussian quadrature
    if not phi_max and not phi_min:
        return result*coeff



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
    #print('result with phi', new_result*coeff)
    return new_result*coeff





