# This module implements all known conversions from DP



import numpy as np
from autodp import utils
import math
from autodp import rdp_bank
from scipy.optimize import minimize_scalar, root_scalar



def puredp_to_rdp(eps):
    # From pure dp to RDP
    assert(eps >= 0)

    def rdp(alpha):
        assert(alpha >= 0)
        if alpha==1:
            # Calculate this by l'Hospital rule
            return eps*(math.cosh(eps)-1)/math.sinh(eps)
        elif np.isinf(alpha):
            return eps
        elif alpha>1:
            # in the proof of Lemma 4 of Bun et al. (2016)
            s, mag = utils.stable_log_diff_exp(utils.stable_log_sinh(alpha*eps),
                                               utils.stable_log_sinh((alpha-1)*eps))
            return (mag - utils.stable_log_sinh(eps))/(alpha-1)
        else:
            return min(alpha * eps * eps /2, eps*(math.cosh(eps)-1)/math.sinh(eps))

    return rdp

def puredp_to_fdp(eps):
    # From Wasserman and Zhou
    def fdp(fpr):
        return np.max(np.array([0, 1-np.exp(eps)*fpr, np.exp(-eps)*(1-fpr)]))
    return fdp

def puredp_to_approxdp(eps):
    # Convert pureDP to approx dp
    # Page 3 of https://eprint.iacr.org/2018/277.pdf
    def approxdp(delta):
        s,mag = utils.stable_log_diff_exp(eps, np.log(delta))
        return mag
    return approxdp

def rdp_to_approxdp(rdp, alpha_max=np.inf, BBGHS_conversion=True):
    # from RDP to approx DP
    # alpha_max is an optional input which sometimes helps avoid numerical issues
    # By default, we are using the RDP to approx-DP conversion due to BBGHS'19's Theorem 21
    # paper: https://arxiv.org/pdf/1905.09982.pdf
    # if you need to use the simpler RDP to approxDP conversion for some reason, turn the flag off

    def approxdp(delta):
        """
        approxdp outputs eps as a function of delta based on rdp calculations

        :param delta:
        :return: the \epsilon with a given delta
        """

        if delta < 0 or delta > 1:
            print("Error! delta is a probability and must be between 0 and 1")
        if delta == 0:
            return rdp(np.inf)
        else:
            def fun(x):  # the input the RDP's \alpha
                if x <= 1:
                    return np.inf
                else:
                    if BBGHS_conversion:
                        return np.maximum(rdp(x) + np.log((x-1)/x)
                                          - (np.log(delta) + np.log(x))/(x-1), 0)
                    else:
                        return np.log(1 / delta) / (x - 1) + rdp(x)

            results = minimize_scalar(fun, method='Brent', bracket=(1,2), bounds=[1, alpha_max])
            if results.success:
                return results.fun
            else:
                # There are cases when certain \delta is not feasible.
                # For example, let p and q be uniform the privacy R.V. is either 0 or \infty and unless all \infty
                # events are taken cared of by \delta, \epsilon cannot be < \infty
                return np.inf
    return approxdp


def single_rdp_to_fdp(alpha, rho):
    assert (alpha >= 0.5)

    def fdp(x):
        assert (0 <= x <= 1)

        if x == 0:
            return 1
        elif x == 1:
            return 0

        if alpha == 1:
            # in this case rho is the KL-divergence
            def fun(y):
                assert (0 <= y <= 1 - x)
                if y == 0:
                    if x == 1:
                        return 0
                    else:
                        return np.inf
                elif y == 1:
                    if x == 0:
                        return 0
                    else:
                        return np.inf
                diff1 = (x * (np.log(x) - np.log(1 - y))
                         + (1 - x) * (np.log(1 - x) - np.log(y)) - rho)
                diff2 = (y * (np.log(y) - np.log(1 - x))
                         + (1 - y) * (np.log(1 - y) - np.log(x)) - rho)

                return np.maximum(diff1, diff2)
        else:
            # This is the general case for Renyi Divergence with \alpha > 1 or \alpha <1
            # find y such that
            # log(x^alpha (1-y)^{1-alpha} + (1-x)^alpha y^{1-alpha}) =  rho(alpha-1))
            # and log(y^alpha (1-x)^{1-alpha}) + (1-y)^alpha x^{1-alpha} =  rho(alpha-1))
            def fun(y):
                if y == 0:
                    if x == 1:
                        return 0
                    else:
                        return np.inf
                elif y == 1:
                    if x == 0:
                        return 0
                    else:
                        return np.inf

                diff1 = (utils.stable_logsumexp_two(alpha * np.log(x) + (1 - alpha) * np.log(1 - y),
                                                    alpha * np.log(1 - x) + (1 - alpha) * np.log(y))
                         - rho * (alpha - 1))
                diff2 = (utils.stable_logsumexp_two(alpha * np.log(y) + (1 - alpha) * np.log(1 - x),
                                                    alpha * np.log(1 - y) + (1 - alpha) * np.log(x))
                         - rho * (alpha - 1))
                if alpha > 1:
                    return np.maximum(diff1, diff2)
                else: # alpha < 1
                    # Notice that the sign of the inequality is toggled
                    return np.minimum(diff1, diff2)


        def normal_equation(y):
            # for finding the root
            return abs(fun(y))

        # there are two roots, we care about the roots smaller than 1-x
        results = minimize_scalar(normal_equation, bounds=[0, 1-x], method='bounded',
                                  options={'xatol':1e-9*(1-x)})
        if results.success:
            return results.x
        else:
            return 0.0
    return fdp


def rdp_to_fdp(rdp, alpha_max=np.inf):
    # RDP function to FDP.
    # maximize the fdp over alpha
    def fdp(x):
        assert (0 <= x <= 1)
        if x == 0:
            return 1
        elif x == 1:
            return 0

        def fun(alpha):
            if alpha < 0.5:
                return np.inf
            else:
                single_fdp = single_rdp_to_fdp(alpha, rdp(alpha))
                return -single_fdp(x)

        # This will use brent to start with 1,2.
        results = minimize_scalar(fun, bracket=(0.5, 2), bounds=(0.5, alpha_max))
        if results.success:
            return -results.fun
        else:
            return 0.0
    return fdp


def single_rdp_to_fdp_and_fdp_grad_log(alpha, rho):
    # Return two functions
    # the first function outputs log(1-fdp(x)) as a function of logx
    # the second function outputs log(-partial fdp(x)) as a function of logx
    # The format of the output of the second function is an interval.
    assert (alpha >= 0.5)

    def diff1_KL(logx,u):
        assert(logx < u < 0)
        return (np.exp(logx) * (logx - u)
                + (1 - np.exp(logx)) * (np.log(1 - np.exp(logx)) - np.log(1-np.exp(u))) - rho)

    def diff2_KL(logx,u):
        return ((1 - np.exp(u)) * (np.log(1 - np.exp(u)) - np.log(1 - np.exp(logx)))
                + np.exp(u) * (u - logx) - rho)

    def diff1_general(logx,u):
        return (utils.stable_logsumexp_two(alpha * logx + (1 - alpha) * u,
                                           alpha * np.log(1 - np.exp(logx))
                                           + (1 - alpha) * np.log(1 - np.exp(u)))
         - rho * (alpha - 1))

    def diff2_general(logx,u):
        return (utils.stable_logsumexp_two(alpha * np.log(1-np.exp(u))
                                           + (1 - alpha) * np.log(1 - np.exp(logx)),
                                           alpha * u + (1 - alpha) * logx) - rho * (alpha - 1))

    def grad1_KL(logx,u):
        mag1 = np.log(u - logx + np.log(1-np.exp(logx)) - np.log(1-np.exp(u)))
        s, mag2 = utils.stable_log_diff_exp(np.log(1-np.exp(logx))- np.log(1-np.exp(u)),logx - u)
        return mag1 - mag2
        # return (logx - u - np.log(1-np.exp(logx))
        #         + np.log(1-np.exp(u))) / ((1-np.exp(logx))/(1-np.exp(u))
        #                                   - np.exp(logx)/np.exp(u))

    def grad2_KL(logx,u):
        mag1 = np.log(u - logx + np.log(1-np.exp(logx)) - np.log(1-np.exp(u)))
        s, mag2 = utils.stable_log_diff_exp(u-logx, np.log(1-np.exp(u))- np.log(1-np.exp(logx)))
        return mag2 - mag1
        # return ((1-np.exp(u))/(1-np.exp(logx))
        #         - np.exp(u)/np.exp(logx)) / (u - logx
        #                                     - np.log(1-np.exp(u)) + np.log(1-np.exp(logx)))

    def grad1_general(logx,u):
        #return - grad1_general(np.log(1-np.exp(u)), np.log(1-np.exp(logx)))

        s, mag = utils.stable_log_diff_exp(alpha * (np.log(1 - np.exp(logx))
                                                    - np.log(1 - np.exp(u))), alpha * (logx-u))
        if alpha > 1:
            s, mag1 = utils.stable_log_diff_exp((alpha-1) * (np.log(1 - np.exp(logx))
                                                             - np.log(1 - np.exp(u))),
                                                (alpha-1) * (logx-u))
            return np.log(alpha)-np.log(alpha-1) + mag1 - mag
        else:
            s, mag1 = utils.stable_log_diff_exp((alpha-1) * (logx-u),
                                                (alpha-1) * (np.log(1 - np.exp(logx))
                                                             - np.log(1 - np.exp(u))))
            return np.log(alpha)-np.log(1-alpha) + mag1 - mag

    def grad2_general(logx,u):
        s, mag = utils.stable_log_diff_exp(alpha * (u - logx),
                                           alpha * (np.log(1 - np.exp(u))
                                                    - np.log(1 - np.exp(logx))))
        if alpha > 1:
            s, mag2 = utils.stable_log_diff_exp((alpha-1) * (u - logx),
                                                (alpha-1) * (np.log(1 - np.exp(u))
                                                - np.log(1 - np.exp(logx))))
            return (np.log(1-1.0/alpha)) + mag - mag2
        else:  # if alpha < 1
            s, mag2 = utils.stable_log_diff_exp((alpha-1) * (np.log(1 - np.exp(u))
                                                - np.log(1 - np.exp(logx))),
                                                (alpha - 1) * (u - logx))
            return np.log(1.0/alpha - 1) + mag - mag2

    def log_one_minus_fdp(logx):
        #assert (0 <= x <= 1)
        assert(logx <= 0)

        if logx == 0: # x==1, f(x) should be 0
            return 0
        elif np.isneginf(logx): # x = 0,  f(x) should be 1
            return -np.inf

        # Now define the non-linear equation ``fun''
        # such that the u such that fun(u) = 0 gives log(1-f(x))

        if alpha == 1:
            # in this case rho is the KL-divergence
            def fun(u):
                assert( u >= logx)
                # assert (0 <= y <= 1 - x)
                if u == 0: #y == 0:
                    if logx == 0: #x == 1:
                        return 0
                    else:
                        return np.inf
                elif np.isneginf(u): #y == 1:
                    if np.isneginf(logx): #x == 0:
                        return 0
                    else:
                        return np.inf

                diff1 = diff1_KL(logx,u)
                #diff1 = (x * (np.log(x) - np.log(1 - y))
                #         + (1 - x) * (np.log(1 - x) - np.log(y)) - rho)
                diff2 = diff2_KL(logx,u)

                #diff2 = (y * (np.log(y) - np.log(1 - x))
                #         + (1 - y) * (np.log(1 - y) - np.log(x)) - rho)

                return np.maximum(diff1, diff2)
        else:
            # This is the general case for Renyi Divergence with \alpha > 1 or \alpha <1
            # find y such that
            # log(x^alpha (1-y)^{1-alpha} + (1-x)^alpha y^{1-alpha}) =  rho(alpha-1))
            # and log(y^alpha (1-x)^{1-alpha}) + (1-y)^alpha x^{1-alpha} =  rho(alpha-1))
            def fun(u):
                assert( u >= logx)
                if u == 0: #y == 0:
                    if logx == 0: #x == 1:
                        return 0
                    else:
                        return np.inf
                elif np.isneginf(u): #y == 1:
                    if np.isneginf(logx): #x == 0:
                        return 0
                    else:
                        return np.inf

                # diff1 = (utils.stable_logsumexp_two(alpha * np.log(x) + (1 - alpha) * np.log(1 - y),
                #                                     alpha * np.log(1 - x) + (1 - alpha) * np.log(y))
                #          - rho * (alpha - 1))
                diff1 = diff1_general(logx,u)
                # diff2 = (utils.stable_logsumexp_two(alpha * np.log(y) + (1 - alpha) * np.log(1 - x),
                #                                     alpha * np.log(1 - y) + (1 - alpha) * np.log(x))
                #          - rho * (alpha - 1))
                diff2 = diff2_general(logx,u)
                if alpha > 1:
                    return np.maximum(diff1, diff2)
                else: # alpha < 1
                    # Notice that the sign of the inequality is toggled
                    return np.minimum(diff1, diff2)

        def normal_equation(u):
            # for finding the root
            return abs(fun(u))

        # there are two roots, we care about the roots smaller than 1-x
        results = minimize_scalar(normal_equation, bounds=[logx,0], method='bounded',
                                  options={'xatol':1e-8})
        if results.success:
            return results.x
        else:
            return 0.0



    def log_neg_partial_fdp(logx):
        assert(logx <= 0)

        if np.isneginf(logx): # x = 0,  the gradient is negative infinity unless alpha = +inf
            # but alpha = +inf won't be passed into here.
            return  np.inf, np.inf
        elif logx == 0: # x = 1
            return 0, 0

        u = log_one_minus_fdp(logx)
        # Find which leg is active, and output the log (- subgradient)
        tol = 1e-5

        grad_l = np.inf
        grad_h = 0

        if alpha == 1:
            err = min(abs(diff1_KL(logx, u)), abs(diff2_KL(logx,u)))
            if err > tol:
                print('no solution found!')

            if abs(diff1_KL(logx,u)) <= tol:
                grad_l = grad1_KL(logx,u)
                grad_h = grad_l
            if abs(diff2_KL(logx,u)) <= tol:
                grad = grad2_KL(logx, u)
                grad_l = min(grad,grad_l)
                grad_h = max(grad,grad_h)
        else:
            err = min(abs(diff1_general(logx, u)), abs(diff2_general(logx,u)))
            if err > tol:
                print('no solution found!')
            if abs(diff1_general(logx,u)) <= tol:
                grad_l = grad1_general(logx,u)
                grad_h = grad_l
            if abs(diff2_general(logx,u)) <= tol:
                grad = grad2_general(logx,u)
                grad_l = min(grad,grad_l)
                grad_h = max(grad,grad_h)
        return [grad_l,grad_h]


    # ------------ debugging --------------------------
    # def fdp(x):
    #     return 1- np.exp(log_one_minus_fdp(np.log(x)))
    #
    # fdp_ref = single_rdp_to_fdp(alpha, rho)
    #
    # import matplotlib.pyplot as plt
    #
    # fpr_list = np.linspace(0,1,100)
    # plt.figure(1)
    # fnr1 = [fdp(x) for x in fpr_list]
    # fnr2 = [fdp_ref(x) for x in fpr_list]
    # plt.plot(fpr_list, fnr1)
    # plt.plot(fpr_list, fnr2)
    #
    # x = 0.01
    # u = log_one_minus_fdp(np.log(x))
    # log_minus_grad_ref = grad2_general(np.log(x), u)
    #
    # log_minus_grad = log_neg_partial_fdp(np.log(x))
    #
    #
    # def grad2_general_new(x,y):
    #     return - alpha*((x/(1-y))**(alpha-1) - ((1-x)/y)**(alpha-1)) / (1-alpha) / ( - (x/(1-y))**alpha + ((1-x)/y)**alpha)
    #
    # def Fxy(x,y):
    #     return (x/(1-y))**alpha * (1-y) +  ((1-x)/y)**alpha * y - np.exp((alpha-1)*rho)
    #
    # y = 1-np.exp(u)
    #
    # grad_ref = grad2_general_new(x,y)
    #
    # grad = -np.exp(log_minus_grad)
    # grad = grad[0]
    #
    # def tangent_line(v):
    #     return y + grad * (v - x)
    #
    # plt.plot(fpr_list, tangent_line(fpr_list))
    #
    # plt.ylim([0,1])
    # plt.xlim([0,1])
    #
    # plt.show()


    return log_one_minus_fdp, log_neg_partial_fdp


def rdp_to_fdp_and_fdp_grad_log(rdp, alpha_max=np.inf):
    # Return the a function that outputs the minimum of
    # log(1-fdp_alpha(x)) and the corresponding log(-partial fdp_alpha(x)) at the optimal \alpha.

    # This, when plugged into the standard machinery, would allow a more direct conversion from RDP.

    def log_one_minus_fdp(logx):
        assert (logx <= 0)
        if np.isneginf(logx):# x == 0:
            return [-np.inf, np.inf] # y = 1,  log (1-y) = -np.inf,  alpha is inf (pure DP)
        elif logx == 0:
            return [0, np.inf]

        def fun(alpha):
            if alpha < 0.5:
                return np.inf
            else:
                log_one_minus_fdp_alpha, tmp = single_rdp_to_fdp_and_fdp_grad_log(alpha, rdp(alpha))
                return log_one_minus_fdp_alpha(logx)

        # This will use brent to start with 1,2.
        results = minimize_scalar(fun, bracket=(0.5, 2), bounds=(0.5, alpha_max))
        if results.success:
            return [results.fun, results.x]
        else:
            return [log_one_minus_fdp(results.x), results.x]

    def log_one_minus_fdp_only(logx):
        res = log_one_minus_fdp(logx)
        return res[0]

    def log_neg_partial_fdp(logx):
        assert (logx <=0)
        if np.isneginf(logx):# x == 0:
            tmp = rdp(np.inf)
            return [tmp, np.inf] # y = 1,  log (1-y) = -np.inf,  alpha is inf (pure DP)
        elif logx == 0:
            tmp = rdp(np.inf)
            return [-np.inf, -tmp]

        # The following implements the more generic case
        # when we need to find the alpha that is active

        res = log_one_minus_fdp(logx)
        best_alpha = res[1]
        tmp, log_neg_partial_fdp_alpha = single_rdp_to_fdp_and_fdp_grad_log(best_alpha,
                                                                            rdp(best_alpha))
        return log_neg_partial_fdp_alpha(logx)

    return log_one_minus_fdp_only, log_neg_partial_fdp


def approxdp_to_approxrdp(eps,delta):
    # from a single eps,delta calculation to an approxdp function
    def approxrdp(alpha, delta1):
        if delta1 >= delta:
            rdp = puredp_to_rdp(eps)
            return rdp(alpha)
        else:
            return np.infty
    return approxrdp


def approxdp_func_to_approxrdp(eps_func):
    # from an approximate_dp function to approxrdp function
    def approxrdp(alpha, delta):
        rdp = puredp_to_rdp(eps_func(delta))
        return rdp(alpha)

    return approxrdp



def approxdp_to_fdp(eps, delta):
    # from a single eps, delta approxdp to fdp
    assert(eps >= 0 and 0 <= delta <= 1)

    def fdp(fpr):
        assert(0 <= fpr <= 1)
        if fpr == 0: # deal with log(0) below
            return 1-delta
        elif np.isinf(eps):
            return 0
        else:
            return np.max(np.array([0, 1-delta-np.exp(eps)*fpr, np.exp(-eps)*(1-delta-fpr)]))
    return fdp


def approxdp_func_to_fdp(func, delta_func=False):
    """
    from an approxdp function to fdp
    :param func: epsilon as a function of delta by default.
    :param delta_func: if the flag is True, then 'func' is a delta as a function of epsilon.
    :return: fdp function
    """
    #
    # By default, logdelta_func is False, and func is eps as a function of delta
    # fpr = maximize_{delta} approxdp_to_fdp(eps(delta),delta)(fpr)
    # if delta_func is True, it means that 'func' is a delta as a function of eps, then
    # fpr = maximize_{delta} approxdp_to_fdp(eps,delta(eps))(fpr)
    if delta_func:
        def fdp(fpr):

            assert(0 <= fpr <= 1)
            if fpr == 1:
                return 0

            def fun(eps):
                fdp_eps = approxdp_to_fdp(eps, func(eps))
                fnr = fdp_eps(fpr)
                return -fnr

            results = minimize_scalar(fun, bounds=[0, +np.inf], options={'disp': False})
            if results.success:
                return -results.fun
            else:
                return 0
    else:
        def fdp(fpr):
            assert(0 <= fpr <= 1)
            if fpr == 1:
                return 0

            def fun(delta):
                fdp_delta = approxdp_to_fdp(func(delta), delta)
                fnr = fdp_delta(fpr)
                return -fnr

            results = minimize_scalar(fun, method='Bounded', bounds=[0, 1-fpr],
                                      options={'disp': False})
            if results.success:
                return -results.fun
            else:
                return 0
    return fdp




def fdp_fdp_grad_to_approxdp(fdp, fdp_grad, log_flag = False):
    # when there is a dedicated implementation of fdp_grad

    # If the log flag is False, then
    #  fdp takes x \in [0,1] and output f(x)
    #  fdp_grad takes x \in [0,1] and output the subdifferential as an interval [grad_l, grad_h]


    # If log_flag is True, then it indicates that
    #   1. the first argument denotes log(1-fdp) as a function of logx
    #   2. the second argument denotes log(- partial fdp) as a function of logx

    if log_flag:
        fun1 = fdp
        fun2 = fdp_grad
    else:
        def fun1(logx):
            assert(logx <= 0)
            if np.isneginf(logx): # x == 0
                return np.log(1-fdp(0))
            elif logx == 0:  # x == 1
                return 1.0
            else:
                return np.log(1-fdp(np.exp(logx)))

        def fun2(logx):
            assert(logx <= 0)
            if np.isneginf(logx):
                grad_l, grad_h = fdp_grad(0)
            else:
                grad_l, grad_h = fdp_grad(np.exp(logx))
            log_neg_grad_l = np.log(-grad_l)
            log_neg_grad_h = np.log(-grad_h)

            if log_neg_grad_l > log_neg_grad_h:
                # in case the order is swapped
                tmp = log_neg_grad_h
                log_neg_grad_h = log_neg_grad_l
                log_neg_grad_l = tmp

            return log_neg_grad_l, log_neg_grad_h

    def find_logx(delta):
        def fun(logx):
            if np.isneginf(logx):
                output = np.log(delta) - fun1(logx)
                return output,output
            else:
                log_neg_grad_l, log_neg_grad_h = fun2(logx)
                log_one_minus_f = fun1(logx)
                low = utils.stable_logsumexp_two(log_neg_grad_l + logx,
                                                 np.log(delta)) - log_one_minus_f
                high = utils.stable_logsumexp_two(log_neg_grad_h + logx,
                                                  np.log(delta)) - log_one_minus_f
                return low, high

        def normal_equation(logx):
            if logx > 0:
                return np.inf
            low, high = fun(logx)
            if low <= 0 <= high:
                return 0
            else:
                return min(abs(high),abs(low))

        def normal_equation_loglogx(loglogx):
            logx = np.exp(loglogx)
            return normal_equation(logx)

        # find x such that y = 1-\delta
        tmp = fun1(np.log(1 - delta))
        if abs(tmp) < 1e-5:
            bound1 = np.log(-tmp - tmp**2 / 2 - tmp**3 / 6)
        else:
            bound1 = np.log(1-np.exp(fun1(np.log(1-delta))))
        #results = minimize_scalar(normal_equation, bounds=[-np.inf,0], bracket=[-1,-2])
        results = minimize_scalar(normal_equation, method="Bounded", bounds=[bound1,0],
                                  options={'xatol': 1e-10, 'maxiter': 500, 'disp': 0})
        if results.success:
            if abs(results.fun) > 1e-4 and abs(results.x)>1e-10:
                # This means that we hit xatol (x is close to 0, but
                # the function value is not close to 0) In this case let's do an even larger search.
                raise RuntimeError("'find_logx' fails to find the tangent line.")
            else:
                return results.x
        else:
            raise RuntimeError(f"'find_logx' fails to find the tangent line: {results.message}")

    def approxdp(delta):
        if delta == 0:
            logx = -np.inf
            log_neg_grad_l, log_neg_grad_h = fun2(logx)
            return log_neg_grad_l
        elif delta == 1:
            return 0.0
        else:
            logx = find_logx(delta)
            log_one_minus_f = fun1(logx)
            # log_neg_grad_l, log_neg_grad_h = fun2(logx)
            s, mag = utils.stable_log_diff_exp(log_one_minus_f,np.log(delta))
            eps = mag - logx
            if eps < 0:
                return 0.0
            else:
                return eps

    #approxdp(1e-3)

    return approxdp



    # def findx(delta):
    #
    #     def fun(x):
    #         # if log_flag:
    #         #     if x == 0:
    #         #         return np.log(delta) - fun1(0)
    #         #     else:
    #         #         return utils.stable_logsumexp_two(fun2(x) + np.log(x), np.log(delta)) - fun1(x)
    #         #else:
    #         fx = fdp(x)
    #         if x == 0:
    #             output = np.log(delta) - np.log(1 - fx)
    #             return output
    #         else:
    #             grad_l, grad_h = fdp_grad(x)
    #         return utils.stable_logsumexp_two(np.log(-fdp_grad(x)) + np.log(x),
    #                                           np.log(delta)) - np.log(1-fx)
    #
    #     def normal_equation(x):
    #         return abs(fun(x))
    #     results = minimize_scalar(normal_equation, method="Bounded", bounds=[0,1],
    #                               options={'xatol': min(1e-10,1e-3*delta), 'maxiter': 500, 'disp': 0})
    #     if results.success:
    #         return results.x
    #     else:
    #         return None
    #
    # def approxdp(delta):
    #     x = findx(delta) - min(1e-10,1e-3*delta)
    #     if log_flag:
    #         return fun2(x)
    #     else:
    #         return np.log(-fdp_grad(x))
    # return approxdp



def fdp_to_approxdp(fdp):
    # Check out Proposition 2.12 of Dong, Roth and Su
    # if given a symmetric fdp function f,
    # its convex conjugate with some manipulation defines the \delta as a function of \epsilon.
    # How to calculate convex conjugates? Can we just do numerical computation?
    # How to ensure the symmetry of fdp function f?
    # One way we can go is to define an fdp_bank where we code up the conjugate pairs in analytical form
    # This allows one to more easily convert approxdp function to fdp

    fstar = conjugate(fdp)
    # def delta_from_fdp(eps):
    #     return 1 + fstar(-np.exp(eps))
    #
    # approxdp = numerical_inverse(delta_from_fdp)

    # def neg_log_one_plus_fstar_neg_input(x):
    #     return -np.log(1 + fstar(-x))
    #
    # exp_eps = numerical_inverse(neg_log_one_plus_fstar_neg_input)
    # def approxdp(delta):
    #     return np.log(exp_eps(-np.log(delta)))

    def neg_fstar_neg_input(x):
        return -fstar(-x)

    exp_eps = numerical_inverse(neg_fstar_neg_input,[0,1])
    def approxdp(delta):
        return np.log(exp_eps(1-delta))

    return approxdp


def numerical_inverse(f, bounds=None):
    # of a scalar, monotonic function
    def inv_f(y):
        if bounds:
            if y > bounds[1] or y < bounds[0]:
                raise ValueError(f'y value {y} is out of bounds [{bounds[0]},{bounds[1]}].')

        def fun(x):
            return f(x) - y

        # The domain should be encoded in the definition of f directly.
        def normal_equation(x):
            return abs(fun(x))

        results = minimize_scalar(normal_equation, bounds=[1,np.inf], bracket=[1,2])


        #results = root_scalar(fun, options={'disp': False})
        if results.success:
            return results.x
        else:
            raise RuntimeError(f"Failed to invert function {f} at {y}: {results.message}")

    return inv_f


def approxdp_from_its_inverse(delta_func):
    # Convert delta as a function of epsilon to epsilon as a function of delta
    return numerical_inverse(delta_func, bounds=[0,1])



## Utility functions

def conjugate(f,tol=1e-10):
    # numerically evaluate convex conjugate of a convex function f: [0,1] --> [0,1]
    # domain of the function y is [0,1]
    def fstar(x):
        def fun(y):
            return -y*x + f(y)
        results = minimize_scalar(fun, method='Bounded', bounds=(0, 1),
                                  options={'disp': False,'xatol':tol})
        if results.success:
            return -(results.fun + tol)
            # output an upper bound
        else:
            raise RuntimeError(f"Failed to conjugate function {f} at {x}: {results.message}")
    return fstar


def pointwise_minimum(f1, f2):
    def min_f1_f2(x):
        return np.minimum(f1(x), f2(x))
    return min_f1_f2

def pointwise_minimum_two_arguments(f1, f2):
    def min_f1_f2(x, y):
        return np.minimum(f1(x, y), f2(x, y))
    return min_f1_f2

def pointwise_maximum(f1, f2):
    def max_f1_f2(x):
        return np.maximum(f1(x), f2(x))
    return max_f1_f2
