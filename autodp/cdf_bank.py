"""
The work flow of AFA: https://arxiv.org/abs/2106.08567)
Step 1. composes the log of characteristic functions.
Step 2. To convert back to delta(eps) or eps(delta), we will use numerical inversion to
convert the characteristic function back to CDFs.
This module provides two numerical inversion approaches to implement the second step.
cdf_quad: Implement Levy theorem using Gaussian quadratures (recommend).
cdf_fft: Numerical inversion through FFT (see https://www.tandfonline.com/doi/abs/10.1080/03461238.1975.10405087).
"""
import numpy as np
import math
import time
from scipy.stats import norm
from scipy.integrate import quad
import scipy.integrate as integrate
from scipy.fft import fft


def cdf_approx_fft(log_phi, L, N = 5e6):
    """
     Future work: This function inverts the characteristic function to the CDF using FFT.

     The detailed numerical inversion can be found in Algorithm B in
     https://www.tandfonline.com/doi/abs/10.1080/03461238.1975.10405087. This approach
     does not involve truncation errors.
     We consider privacy loss R.V. is zj: = -b + lam*j, where lam = 2pi/(eta*(2N-1)).


     Args:
        log_phi: the log of characteristic (phi)  function.
        L: Limit for the approximation of the privacy loss distribution integral.
        N: Number of points in FFT (FFT is over 2N-1 points)

    Return:
        A list of CDFs of privacy loss R.V. evaluated between [-L, L]

    """

    # evaluate the j-th privacy loss r.v. zj: = -b + lam*j, where lam = 2pi/(eta*(2N-1)).
    eta = np.pi/L
    N = int(N)
    b = -np.pi / eta
    lam = 2*np.pi/(eta*(2*N-1))
    t_list = [m_hat + 1 - N for m_hat in range(2*N-1)]
    t0 = time.time()
    c_nu = lambda t: (1 - t) * np.cos(np.pi * t) + np.sin(np.pi * t) / np.pi

    # FFT is used for the calculation of the trigonometric sums appearing in the formulas.
    # l == 0 is undefined in the original formula, thus we need to exclude it from the fft.
    def f_phi(l):
        if l == 0:
            return 0
        nu = l*1.0/N
        c_t = c_nu(abs(nu))
        return c_t*np.exp(log_phi(l * eta) - 1.j * eta * b * l) / l

    phi_list = [f_phi(m_hat) for m_hat in t_list]
    fft_res = fft(phi_list)

    fft_norm = [fft_res[j] * np.exp(1.j * (N - 1) * 2.0 / (2. * N - 1) * j * np.pi) for j in range(2 * N - 1)]

    """
    cdf[j] denotes the cdf of log(p/q) when evaluates at zj, zj: = -b + lam*j.
    the range of z is [-pi/eta, pi/eta], the mesh-size on zj is lam. To get a good approximation on cdf,
     we  need eta*N to be a large number.
     
    cdf(z) = 0.5 + eta*z/(2pi) -fft_res[z]
    """
    convert_z = lambda j: b+lam*j
    cdf = [0.5 + eta * convert_z(j) / (2 * np.pi) - 1. / (2 * np.pi * 1.j) * fft_norm[j] for j in range(2 * N - 1)]
    return cdf

def cdf_quad(log_phi, ell, n_quad=700, extra_para=None):
    """
     This function computes the CDF of privacy loss R.V. via Levy theorem.
     https://en.wikipedia.org/wiki/Characteristic_function_%28probability_theory%29#Inversion_formulae
     The integration is implemented through Gaussian quadrature.

     Args:
        log_phi: the log of characteristic (phi)  function.
        ell: the privacy loss RV is evaluated at ell
        extra_para: extra parameters used to describe the privacy loss R.V..

    Return: the CDF of the privacy loss RV when evaluated at ell。
    """

    def qua(t):
        """
        Convert [-1, 1] to an infinite integral.
        """
        new_t = t*1.0/(1-t**2)
        phi_result = [log_phi(x) for x in new_t]
        inte_function = 1.j/new_t * np.exp(-1.j*new_t*ell+phi_result)
        return inte_function
    # n is the maximum sampling point used in Gaussian quadrature, setting it to be >700 is usually very accurate.
    inte_f = lambda t: qua(t) * (1 + t ** 2) / ((1 - t ** 2) ** 2)
    res = integrate.fixed_quad(inte_f, -1.0, 1.0, n =n_quad)

    result = res[0]
    return np.real(result)/(2*np.pi)+0.5
