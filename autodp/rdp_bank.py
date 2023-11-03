"""
A collection of analytical expressions of the Renyi Differential Privacy for popular randomized algorithms.
All expressions in this file takes in a function of the RDP order alpha and output the corresponding RDP.

These are used to create symbolic randomized functions for the RDP accountant to track.

Some of the functions contain the renyi divergence of two given distributions, these are useful to keep track of
the per-instance RDP associated with two given data sets.

TO CONTRIBUTORS:  1. any new addition to the rdp_bank should include a reference to the mechanism of
    interesting and the derivation of its RDP.
    2. You should try providing an implementation fo the entire range of alpha >0. If you do not
       have alpha <1, feel free to use any upper bound, e.g., the bound for alpha = 1.

"""


import numpy as np
import math
from autodp import utils
from autodp.utils import _log1mexp


def stable_log_diff_exp(x):
    # ensure that y > x
    # this function returns the stable version of log(exp(y)-exp(x)) if y > x

    mag = np.log(1 - np.exp(x - 0))

    return mag


def RDP_gaussian(params, alpha):
    """
    Args:
        sigma: normalized noise level: std divided by global L2 sensitivity
        alpha: The order of the Renyi Divergence

    Return: Evaluation of the RDP's epsilon
    """
    sigma = params['sigma']
    assert(sigma > 0)
    assert(alpha >= 0)
    return 0.5 / sigma ** 2 * alpha


def RDP_laplace(params, alpha):
    """
    Args:
        b: the ratio of the scale parameter and L1 sensitivity
        alpha: The order of the Renyi Divergence
    Return: Evaluation of the RDP's epsilon
    """

    b = params['b']
    # assert(b > 0)
    # assert(alpha >= 0)
    alpha=1.0*alpha

    if np.isinf(alpha):
        return 1/b
    elif alpha == 1:
        # KL-divergence
        return 1 / b + np.exp(-1 / b) - 1
    elif alpha > 1:  # alpha > 1
        return utils.stable_logsumexp_two((alpha-1.0) / b + np.log(alpha / (2.0 * alpha - 1)),
                                           -1.0*alpha / b + np.log((alpha-1.0) / (2.0 * alpha - 1)))/(alpha-1)
    elif alpha == 0.5:
        return -2*(-1.0/(2*b) + np.log(1 + 1.0/(2*b)))#   -2*np.log(np.exp(-1.0/(2*b))*(1+1.0/(2*b)))
    else:
        return np.log(alpha/(2.0*alpha-1)*np.exp((alpha-1.0)/b) + (alpha-1.0)/(2.0*alpha-1)*np.exp(-1.0*alpha/b))/(alpha-1)
        # Handling the case when alpha = 1/2?

def RDP_zCDP(params,alpha):
    """
    This function implements the RDP of (xi,rho)-zCDP mechanisms.
    Definition 11 of https://arxiv.org/pdf/1605.02065.pdf
    (Extended to  alpha > 0; may need to check for your mechanism)
    :param params: rho --- zCDP parameter
    xi ---- optional zCDP parameter
    :param alpha: The order of the Renyi-Divergence
    :return: the implied RDP at level alpha
    """
    rho = params['rho']
    if 'xi' in params.keys():
        xi = params['xi']
    else:
        xi = 0
    assert (alpha >= 0)
    return xi + rho*alpha

def RDP_independent_noisy_screen(params, alpha):

    """
    The data-independent RDP of ``Noisy Screening" (Theorem 7 in Private-kNN).

    The method is described in https://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_Private-kNN_Practical_Differential_Privacy_for_Computer_Vision_CVPR_2020_paper.html)
    The exact data-independent bound requires searching a max_count from [k/c, k] to maximize RDP_noisy_screening for any alpha

    Args:
        params: contains three parameters. params['thresh'] is the threshold for noisy screening,
        k is the number of neighbors in Private-kNN, sigma is the noisy scale.

    Returns:
        The RDP of data-independent noisy screening.
    """
    threshold =params['thresh']
    k = params['k']
    sigma = params['sigma']
    import scipy.stats
    rdp = []

    for adjacent in [+1, -1]:
        for max_count in [int(k/10),threshold]:

            logp = scipy.stats.norm.logsf(threshold - max_count, scale=sigma)
            logq = scipy.stats.norm.logsf(threshold - max_count+adjacent, scale=sigma)
            log1q = _log1mexp(logq)
            log1p = _log1mexp(logp)
            if alpha == 1:
                return np.exp(logp)*(logp-logq) +(1-np.exp(logp))*(log1p-log1q)
            elif np.isinf(alpha):
                return np.abs(np.exp(logp-logq)*1.0)

            term1 = alpha*logp -(alpha-1)*logq
            term2 = alpha*log1p -(alpha-1)*log1q
            log_term = utils.stable_logsumexp_two(term1,term2)
            rdp.append(log_term)
    log_term = np.max(log_term)
    return 1.0*log_term /(alpha-1)

def RDP_noisy_screen(params, alpha):

    """
    return the data-dependent RDP of ``Noisy Screening" (Theorem 7 in Private-kNN).

    Args:
        params contains two parameters logp and logq. logp denotes the log of
         probability of P(max_vote + noise > Threshold).

    """
    logp =params['logp']
    logq = params['logq']
    log1q = _log1mexp(logq)
    log1p = _log1mexp(logp)
    if alpha == 1:
        return np.exp(logp)*(logp-logq) +(1-np.exp(logp))*(log1p-log1q)
    elif np.isinf(alpha):
        return np.abs(np.exp(logp-logq)*1.0)

    term1 = alpha*logp -(alpha-1)*logq
    term2 = alpha*log1p -(alpha-1)*log1q
    log_term = utils.stable_logsumexp_two(term1,term2)

    return 1.0*log_term /(alpha-1)

def RDP_inde_pate_gaussian(params, alpha):
    """
    Return the data-independent RDP of Noisy Aggregation (the global sensitivity is 2).
    The method is descriped in https://arxiv.org/abs/1802.08908

    Args:
        sigma: noisy scale added to the vote count.

    """
    sigma = params['sigma']
    return 1.0/sigma **2 *alpha

def RDP_depend_pate_gaussian(params, alpha):

    """
    Return the data-dependent RDP of GNMAX (proposed in PATE2)
    Bounds RDP from above of GNMax given an upper bound on q (Theorem 6).

    Args:
      logq: Natural logarithm of the probability of a non-argmax outcome.
      sigma: Standard deviation of Gaussian noise.
      orders: An array_like list of Renyi orders.

    Returns:
      Upper bound on RPD for all orders. A scalar if orders is a scalar.

    Raises:
      ValueError: If the input is malformed.
    """
    logq = params['logq']
    sigma = params['sigma']

    if alpha == 1:
        p = np.exp(logq)
        w = (2 * p - 1) * (logq - _log1mexp(logq))
        return w
    if logq > 0 or sigma < 0 or np.any(alpha < 1):  # not defined for alpha=1
        raise ValueError("Inputs are malformed.")

    if np.isneginf(logq):  # If the mechanism's output is fixed, it has 0-DP.
        print('isneginf', logq)
        if np.isscalar(alpha):
            return 0.
        else:
            return np.full_like(alpha, 0., dtype=np.float)

    variance = sigma ** 2

    # Use two different higher orders: mu_hi1 and mu_hi2 computed according to
    # Proposition 10.
    mu_hi2 = math.sqrt(variance * -logq)
    mu_hi1 = mu_hi2 + 1

    orders_vec = np.atleast_1d(alpha)

    ret = orders_vec / variance  # baseline: data-independent bound

    # Filter out entries where data-dependent bound does not apply.
    mask = np.logical_and(mu_hi1 > orders_vec, mu_hi2 > 1)

    rdp_hi1 = mu_hi1 / variance
    rdp_hi2 = mu_hi2 / variance

    log_a2 = (mu_hi2 - 1) * rdp_hi2

    # Make sure q is in the increasing wrt q range and A is positive.
    if (np.any(mask) and logq <= log_a2 - mu_hi2 *
            (math.log(1 + 1 / (mu_hi1 - 1)) + math.log(1 + 1 / (mu_hi2 - 1))) and
            -logq > rdp_hi2):
        # Use log1p(x) = log(1 + x) to avoid catastrophic cancellations when x ~ 0.
        log1q = _log1mexp(logq)  # log1q = log(1-q)
        log_a = (alpha - 1) * (
                log1q - _log1mexp((logq + rdp_hi2) * (1 - 1 / mu_hi2)))
        log_b = (alpha - 1) * (rdp_hi1 - logq / (mu_hi1 - 1))

        # Use logaddexp(x, y) = log(e^x + e^y) to avoid overflow for large x, y.
        log_s1 = utils.stable_logsumexp_two(log1q + log_a, logq + log_b)
        log_s = np.logaddexp(log1q + log_a, logq + log_b)
        ret[mask] = np.minimum(ret, log_s / (alpha - 1))[mask]
    # print('alpha ={} mask {}'.format(alpha,ret))
    if ret[mask] < 0:
        print('negative ret', ret)
        print('log_s1 ={} log_s = {}'.format(log_s1, log_s))
        print('alpha = {} mu_hi1 ={}'.format(alpha, mu_hi1))
        print('log1q = {} log_a = {} log_b={} log_s = {}'.format(log1q, log_a, log_b, log_s))
        ret[mask] = 1. / (sigma ** 2) * alpha
        # print('replace ret with', ret)
    assert np.all(ret >= 0)

    if np.isscalar(alpha):
        return np.asscalar(ret)
    else:
        return ret


def RDP_randresponse(params, alpha):
    """
    The RDP of the random response mechanism.

    Args:
        p: is the Bernoulli probability p of outputting the truth
        alpha: The order of the Renyi Divergence

    :return: Evaluation of the RDP's epsilon
    """

    p = params['p']
    assert((p >= 0) and (p <= 1))
    # assert(alpha >= 0)
    if p == 1 or p == 0:
        return np.inf
    if alpha == 1:
        return (2 * p - 1) * np.log(p / (1 - p))
    elif np.isinf(alpha):
        return np.abs(np.log((1.0*p/(1-p))))
    else:  # alpha > 1
        return utils.stable_logsumexp_two(alpha * np.log(p) + (1 - alpha) * np.log(1 - p),
                                           alpha * np.log(1 - p) + (1 - alpha) * np.log(p))/(alpha-1)



def RDP_expfamily(params, alpha):
    """
    The RDP of the exponential mechanism.
    The details can be found in Proposition 29 and Remark 30 of Wang, Balle, Kasiviswanathan (2018).
    Args:
        Delta: max distance of the natural parameters between two adjacent data sets in a certain norms.
        'L' 'B' are lambda functions. They are upper bounds of the local smoothness and local Lipschitzness
        of the log-partition function A, as a function of the radius of the local neighborhood in that norm.
        alpha: The order of the Renyi Divergence

    """
    Delta = params['Delta']
    L  = params['L'] # Local smoothness function that takes radius kappa as a input.
    B = params['B']  # Local Lipschitz function that takes radius kappa as a input.

    return np.minimum(alpha * L(alpha*Delta) * Delta ** 2,
                      (B((alpha-1)*Delta) + B(Delta)) * Delta)


def pRDP_diag_gaussian(params, alpha):
    """
    :param params:
        'mu1', 'mu2', 'sigma1', 'sigma2', they are all d-dimensional numpy arrays
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the pRDP's epsilon

    See page 27 of http://mast.queensu.ca/~communications/Papers/gil-msc11.pdf for reference.
    """
    # the parameter is mu1, mu2, sigma1, sigma2
    # they are all d-dimensional numpy arrays
    # Everything can be generalized to general covariance, but
    # calculating A and checking the positive semidefiniteness of A is tricky

    mu1 = params['mu1']
    mu2 = params['mu2']
    sigma1 = params['sigma1']
    sigma2 = params['sigma2']

    def extrapolate(a, b):
        return alpha * a +(1 - alpha) * b

    #A = extrapolate(sigma1 ** (-1), sigma2 ** (-1))

    sigma = extrapolate(sigma2, sigma1)

    if not (sigma > 0).all():
        return np.inf
    else:
        #sigma = extrapolate(sigma2, sigma1)
        return (alpha / 2.0 * np.dot((mu1 - mu2),  (mu1 - mu2) / sigma) - 0.5 / (alpha-1) *
                (np.sum(np.log(sigma)) - extrapolate(np.sum(np.log(sigma2)), np.sum(np.log(sigma1)))))



def RDP_svt_laplace(params, alpha):
    """
    The RDP of Laplace-based SVT.
    Args:
        params['b']: the Laplace noise scale divide the sensitivity.
        params['k']: the SVT algorithm stops either k queries is achieved
        or the cut-off c is achieved.

    """
    b = params['b']
    k = params['k']  # the algorithm stops either k is achieved or c is achieved
    c = max(params['c'], 1)

    alpha = 1.0 * alpha
    if alpha <= 1:
        eps_1 = (1 / b + np.exp(-1 / b) - 1)
    elif np.isinf(alpha):
        eps_1 = 1 / b
    else:  # alpha > 1
        eps_1 = utils.stable_logsumexp_two((alpha - 1.0) / b + np.log(alpha / (2.0 * alpha - 1)),
                                           -1.0 * alpha / b + np.log((alpha - 1.0) / (2.0 * alpha - 1))) / (alpha - 1)

    eps_2 = 1 / b  # infinity rdp on nu
    c_log_n_c = c * np.log(k / c)
    tilde_eps = eps_2 * (c + 1)  # eps_infinity
    ret_rdp = min(c * eps_2 + eps_1, c_log_n_c * 1.0 / (alpha - 1) + eps_1 * (c + 1))
    ret_rdp = min(ret_rdp, 0.5 * alpha * tilde_eps ** 2)
    if np.isinf(alpha) or alpha == 1:
        return ret_rdp
    # The following is sinh-based method
    tilde_eps = eps_2 * (c + 1)
    cdp_bound = np.sinh(alpha * tilde_eps) - np.sinh((alpha - 1) * tilde_eps)
    cdp_bound = cdp_bound / np.sinh(tilde_eps)
    cdp_bound = 1.0 / (alpha - 1) * np.log(cdp_bound)
    return min(ret_rdp, cdp_bound)


def RDP_gaussian_svt_cgreater1(params, alpha):
    """
    The RDP of the gaussian-based SVT with the cut-off parameter c>1.

    Args:
        k:the maximum length before svt stops
        sigma: noise added to the threshold.
        c: the cut-off parameter in SVT.
    """
    sigma = params['sigma']
    c = max(params['c'], 1)
    k = params['k']  # the algorithm stops either k is achieved or c is achieved
    rdp_rho = 0.5 / (sigma ** 2) * alpha
    c_log_n_c = c * np.log(k / c)
    ret_rdp = c_log_n_c * 1.0 / (alpha - 1) + rdp_rho * (c + 1)
    return ret_rdp


def RDP_gaussian_svt_c1(params, alpha):
    """
    The RDP of the gaussian-based SVT with the cut-off parameter c=1.

    The detailed algorithm is described in Theorem 8 in
    https://papers.nips.cc/paper/2020/file/e9bf14a419d77534105016f5ec122d62-Paper.pdf

    A special case, where $k$ is not required, occurs when:
    1. Queries are non-negative, and have low-sensitivity (assumed),
    2. $\sigma_2 \geq \sqrt{3} \sigma_1$,
    3. $\gamma = 2$
    
    Calculates in the standard bound and the special case (Proposition 10) when applicable.

    params args:
        k (int, optional): the maximum length before svt stops (used in Theorem 8. -- general case)
        sigma (float): noise added to the threshold ($\sigma_1$ ($\sigma_{\\rho}$) in [ZW2020]).
        sigma_nu (float): noise added to each query ($\sigma_2$ ($\sigma_{\nu}$) in [ZW2020]).
            Defaults to \sqrt{3} \sigma_{1}
        Delta (float): global sensitivity for each query
        margin (float): query threshold parameter ($T$ in [ZW2020])
        gamma (float): positive multiplicative factor (default=2)
    """
    # API expects `sigma` to be defined for all gaussian mechanisms, let sigma=sigma_1:
    sigma_1 = params['sigma']
    # [ZW2020] denotes sigma_nu as sigma_2. By default, calibrate query noise to apply Prop. 10:
    sigma_2 = params.get('sigma_nu', np.sqrt(3) * sigma_1)
    # AutoDP uses sigma2 as a param, for consistency we support it
    sigma_2 = params.get('sigma2', sigma_2)
    margin = params.get('margin', 0.)
    Delta = params.get('Delta', 1.)
    gamma = params.get('gamma', 2.)

    rdp_rho = alpha * Delta**2 / (sigma_1 ** 2)
    # By Theorem 8, $\eps_{\nu}(\alpha)$ is upper bounded by 2 \Delta.
    rdp_nu = 2 * alpha * Delta**2 / (sigma_2 **2 )

    # If $k$ provided, compute the standard bound
    ret_rdp = np.inf
    if 'k' in params:
        k = params['k']
        ret_rdp = np.divide(np.log(k) , alpha - 1, where=alpha!=1) + rdp_rho + rdp_nu
    
    # Check if the conditions for the Propisition 10 are satisfied:
    if sigma_2 < np.sqrt(gamma + 1) * sigma_1:
        # Check if k is defined, if not, no bound can be applied correctly
        assert 'k' in params, "$k$ must be defined if Proposition 10 is not applicable."
        return ret_rdp

    # Proposition 10. Bound the stopping time by \E[\E[K | \rho=z]^{\gamma}], where K is a random stopping time
    stop_time_log_part = np.sqrt(1 + gamma)
    stop_time_log_part = stop_time_log_part * np.power(np.sqrt(2 * np.pi), gamma)
    stop_time_log_part = stop_time_log_part * ( np.power(( (margin * (1 + gamma))/sigma_1), gamma ) + 1)
    stop_time_log_part = np.log(stop_time_log_part)
    stop_time_exp_part = (gamma * margin**2) / (2 * sigma_1**2)

    # note exp(0) = 1, which is implicitly added
    moment_term = utils.stable_logsumexp_two(0, stop_time_log_part + stop_time_exp_part)
    moment_term = moment_term / (gamma * (alpha - 1))

    moment_based_bound = rdp_rho + rdp_nu + moment_term
    return min(moment_based_bound, ret_rdp)



def RDP_pureDP(params,alpha):
    """
    This function generically converts pure DP to Renyi DP.
    It implements Lemma 1.4 of Bun et al.'s CDP paper.
    With an additional cap at eps.

    :param params: pure DP parameter
    :param alpha: The order of the Renyi Divergence
    :return:Evaluation of the RDP's epsilon
    """
    eps = params['eps']
    # assert(eps>=0)
    # if alpha < 1:
    #     # Pure DP needs to have identical support, thus - log(q(p>0)) = 0.
    #     return 0
    # else:
    #     return np.minimum(eps,alpha*eps*eps/2)

    assert (alpha >= 0)
    if alpha == 1:
        # Calculate this by l'Hospital rule
        return eps * (math.cosh(eps) - 1) / math.sinh(eps)
    elif np.isinf(alpha):
        return eps
    elif alpha > 1:
        # in the proof of Lemma 4 of Bun et al. (2016)
        s, mag = utils.stable_log_diff_exp(utils.stable_log_sinh(alpha * eps),
                                           utils.stable_log_sinh((alpha - 1) * eps))
        return (mag - utils.stable_log_sinh(eps)) / (alpha - 1)
    else:
        return min(alpha * eps * eps / 2, eps * (math.cosh(eps) - 1) / math.sinh(eps))


def RDP_zCDP(params,alpha):
    """
    This function implements the RDP of (xi,rho)-zCDP mechanisms.
    Definition 11 of https://arxiv.org/pdf/1605.02065.pdf

    (Extended to  alpha > 0; may need to check for your mechanism)


    :param params: rho --- zCDP parameter
    xi ---- optional zCDP parameter
    :param alpha: The order of the Renyi-Divergence
    :return: the implied RDP at level alpha
    """
    rho = params['rho']
    if 'xi' in params.keys():
        xi = params['xi']
    else:
        xi = 0
    assert (alpha >= 0)
    return xi + rho*alpha


def RDP_truncatedCDP(params,alpha):
    """
    This function implements the RDP of (rho,w)-tCDP mechanisms.
    See Definition 1 of
    https://projects.iq.harvard.edu/files/privacytools/files/bun_mark_composable_.pdf

    (Extended to  alpha > 0; may need to check for your mechanism)

    :param params: rho, w
    :param alpha: The order of the Renyi-Divergence
    :return: the implied RDP at level alpha
    """
    rho = params['rho']
    w = params['w']
    assert (alpha >= 0)
    if alpha < w:
        return rho*alpha
    else:
        return np.inf



def RDP_subsampled_pureDP(params, alpha):
    """
    The special function for approximating the privacy amplification by subsampling for pure DP mechanism.
    1. This is more or less tight
    2. This applies for both Poisson subsampling (with add-remove DP definition)
       and Subsampling (with replace DP Definition)
    3. It evaluates in O(1)

    :param params: pure DP parameter, (optional) second order RDP, and alpha.
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the RDP's epsilon

    """
    eps = params['eps']
    #eps2 = params['eps2'] # this parameter is optional, if unknown just use min(eps,2*eps*eps/2)
    if 'eps2' in params.keys():
        eps2= params['eps2']
    else:
        eps2 = np.minimum(eps,eps*eps)
    prob = params['prob']
    assert((prob<1) and (prob >= 0))
    assert(eps >= 0 and eps2 >=0)


    def rdp_int(x):
        if x == np.inf:
            return eps
        s, mag = utils.stable_log_diff_exp(eps,0)
        s, mag2 = utils.stable_log_diff_exp(eps2,0)

        s, mag3 = utils.stable_log_diff_exp(x*utils.stable_logsumexp_two(np.log(1-prob),np.log(prob)+eps),
                                            np.log(x) + np.log(prob) + mag)
        s, mag4 = utils.stable_log_diff_exp(mag3, np.log(1.0*x/2)+np.log(x-1)+2*np.log(prob)
                                            + np.log( np.exp(2*mag) - np.exp(np.min([mag,2*mag,mag2]))))

        return 1/(x-1)*mag4


        ## The following is the implementation of the second line.
        # if x <= 2:
        #     # Just return rdp 2
        #     return utils.stable_logsumexp([0, np.log(1.0*2/2)+np.log(2-1)+2*np.log(prob)
        #                                    + np.min([mag2,mag,2*mag])])
        # else:
        #     return 1/(x-1)*utils.stable_logsumexp([0, np.log(1.0*x/2)+np.log(x-1)+2*np.log(prob)
        #                                            + np.min([mag2,mag,2*mag]),
        #                                           3*np.log(prob) + 3*mag + np.log(x) + np.log(x-1)
        #                                           + np.log(x-2) - np.log(6) +
        #                                           (x-3) * utils.stable_logsumexp_two(np.log(1-prob),np.log(prob)+eps)])

    if alpha < 1:
        return 0
    else:
        return utils.RDP_linear_interpolation(rdp_int, alpha)


def pRDP_asymp_subsampled_gaussian(params, alpha):
    """
    :param params:
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the pRDP's epsilon

    See Example 19 of Wang, Balle, Kasiviswanathan (2018)
    """
    sigma = params['sigma']
    prob = params['prob']
    assert((prob<1) and (prob >= 0))

    # The example where we have an approximately worst case asymptotic data set
    thresh = sigma**2/(prob*(1-prob)) + 1
    if alpha <= 1:
        return 0
    elif alpha >= thresh:
        return np.inf
    else:
        return (prob ** 2 / (2*sigma**2) * alpha * (thresh-1) / (thresh - alpha)
                + np.log((thresh-1)/thresh) / 2
                ) + np.log((thresh-1) / (thresh - alpha)) / 2 /(alpha-1)


def pRDP_asymp_subsampled_gaussian_best_case(params, alpha):
    """
    :param params:
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the pRDP's epsilon

    See Example 20 of Wang, Balle, Kasiviswanathan (2018)
    """
    sigma = params['sigma']
    prob = params['prob']
    n = params['n']
    assert((prob<1) and (prob >= 0))

    # The example where we have an approximately best case data set
    return prob**2 / (2*sigma**2 + prob*(1-prob)*(n-1.0/n)/2) * alpha



def pRDP_expfamily(params, alpha):
    """
    :param params:
        'eta1', 'eta2' are the natural parameters of the exp family distributions.
        'A' is a lambda function handle the log partition. `A' needs to handle cases of infinity.
        'mu' is the mean of the sufficient statistics from distribution 1.
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the pRDP's epsilon
    """
    # Theorem 1.2.19 of http://mast.queensu.ca/~communications/Papers/gil-msc11.pdf
    eta1 = params['eta1']
    eta2 = params['eta2']
    A = params['A']
    mu = params['mu'] # This is used only for calculating KL divergence
    # mu is also the gradient of A at eta1.


    assert(alpha >= 1)

    if alpha == 1:
        return np.dot(eta1-eta2, mu) + A(eta1) - A(eta2)

    def extrapolate(a, b):
        return alpha * a + (1 - alpha) * b

    tmp = A(extrapolate(eta1, eta2))
    if np.isinf(tmp) or np.isnan(tmp):
        return np.inf
    else: # alpha > 1
        return (A(extrapolate(eta1, eta2)) - extrapolate(A(eta1), A(eta2))) / (alpha - 1)
