# This module implements f-DP function templates for various mechanisms
# Note that f-DP is equivalent to the family of (eps,delta)-DP described by the functions
#  delta(eps) or eps(delta), generic conversion is available from the converter module
# however it is numerically better if we implement and support cases when fdp is natural

import numpy as np
from scipy.stats import norm
from autodp import utils


def fDP_gaussian(params, fpr):
    """
    :param params:
        'sigma' --- is the normalized noise level: std divided by global L2 sensitivity
    :param fpr: False positive rate --- input to the fDP function
    :return: Evaluation of the fnr lower bound supported by the gaussian mechanism
    """
    sigma = params['sigma']
    # assert(sigma > 0)
    assert(sigma >= 0)
    if sigma == 0:
        return 0
    else:
        return norm.cdf(norm.ppf(1-fpr)-1/sigma)


def fDP_approx_DP(params, fpr):
    """
    :param params:
        'eps' --- DP parameter 1
        'delta' --- DP parameter 2
    :param fpr: False positive rate --- input to the fDP function
    :return:  Evaluation of the fnr lower bound supported by any pure DP mechanism
    """
    eps = params['eps']
    delta = params['delta']
    assert(eps >= 0)
    if np.isinf(eps):
        return 0
    elif eps == 0:
        return 1-fpr - delta
    else:
        return np.max([0, 1-delta - np.exp(eps)*fpr, np.exp(-eps)(1-delta - fpr)])


def fDP_pure_DP(params, fpr):
    """
    :param params:
        'eps' --- DP parameter
    :param fpr: False positive rate --- input to the fDP function
    :return:  Evaluation of the fnr lower bound supported by any pure DP mechanism
    """
    params['delta'] = 0
    return fDP_approx_DP(params, fpr)


def fdp_grad_gaussian(params,fpr):
    """
    :param params:
        'sigma' --- is the normalized noise level: std divided by global L2 sensitivity
    :param fpr: False positive rate --- input to the fDP function
    :return: Evaluation of derivative of the Tradeoff function at input fpr
    """
    sigma = params['sigma']
    # assert(sigma > 0)
    assert(sigma >= 0)
    if sigma == 0:
        return 0
    else:
        return -norm.pdf(norm.ppf(1-fpr)-1/sigma)/norm.pdf(norm.ppf(1-fpr))

def log_one_minus_fdp_gaussian(params, logfpr):
    """
    :param params:
        'sigma' --- is the normalized noise level: std divided by global L2 sensitivity
    :param logfpr: log of False positive rate --- input to the fDP function
    :return: log(1-f(x)).
    """
    sigma = params['sigma']
    # assert(sigma > 0)
    assert(sigma >= 0)
    if sigma == 0:
        return 0
    else:
        if np.isneginf(logfpr):
            return -np.inf
        else:

            norm_ppf_one_minus_fpr = utils.stable_norm_ppf_one_minus_x(logfpr)

            return norm.logsf(norm_ppf_one_minus_fpr-1/sigma)




def log_neg_fdp_grad_gaussian(params, logfpr):
    """
    :param params:
        'sigma' --- is the normalized noise level: std divided by global L2 sensitivity
    :param logfpr: log of False positive rate --- input to the fDP function
    :return: log(-partial f(x))
    """
    sigma = params['sigma']
    # assert(sigma > 0)
    assert(sigma >= 0)
    if sigma == 0:
        return 0
    else:
        if np.isneginf(logfpr):  # == 0:
            return np.inf, np.inf
        elif logfpr == 0:  #fpr == 1:
            return -np.inf, -np.inf
        else:
            norm_ppf_one_minus_fpr = utils.stable_norm_ppf_one_minus_x(logfpr)
            grad = -(norm_ppf_one_minus_fpr
                     - 1 / sigma) ** 2 / 2 + norm_ppf_one_minus_fpr ** 2 / 2
            return grad, grad
