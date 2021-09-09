"""
Core classes for autodp:

Mechanism --- A `mechanism' describes a randomized algorithm and its privacy properties.
                All `mechanism's (e.g., those in the `mechanism_zoo' module) inherit this class.

Transformer ---  A transformer takes one or a list of mechanism and outputs another mechanism
                All `transformer's (e.g., those in the `transformer_zoo' module, e.g., amplificaiton
                 by sampling, shuffling, and composition) inherit this class.

Calibrator --- A `calibrator' takes a mechanism with parameters (e.g. noise level) and automatically
                choose those parameters to achieve a prespecified privacy budget.
                All `calibrator's (e.g., the Analytical Gaussian Mechanism calibration, and others
                in `calibrator_zoo'inherit this class)

"""

import numpy as np
from autodp import converter


class Mechanism():
    """
     The base mechanism will use typically two functions to describe the mechanism

    # Attributes (actually functions as well):
    # 1: Approximate DP:   epsilon as a function of delta
    # 2. Renyi DP:   RDP epsilon as a function of \alpha
    # 3. Approximate RDP:  approximate RDP. RDP conditioning on a failure probability delta0.
    # 4. f-DP:  Type II error as a function of Type I error. You can get that from Approximate-DP
    #           or FDP directly.
    # 5. epsilon:  Pure DP bound.  If not infinity, then the mechanism satisfies pure DP.
    # 6. delta0:  Failure probability which documents the delta to use for approximate RDP
    #             in the case when there are no information available about the failure event.
    # 7. local_flag: Indicates whether the guarantees are intended to be for local differential privacy
    # 8. group_size: Integer measuring the granuality of DP.  Default is 1.
    # 9. replace_one: Flag indicating whether this is for add-remove definition of DP
    #                 or replace-one version of DP,  default is False

    # CDP and approximate-CDP will be subsumed in RDP bounds
    #
    # If we specify RDP only then it will propagate the RDP calculations to approximate-DP
    # and to f-DP
    # If we specify pure-DP only then it propagates to RDP,  Approximate-DP, f-DP and so on.
    # If we specify approximate-DP only, then it implies an approximate RDP bound with \delta_0.
    # If we specify f-DP only then it propagates to other specifications.
    # If we specify multiple calculations, then it will take the minimum of all of them
    #      in each category
    """


    def __init__(self):
        # Initialize everything with trivial (non-private) defaults
        def RenyiDP(alpha):
            return np.inf

        def approxRDP(delta, alpha):
            return np.inf

        def approxDP(delta):
            return np.inf

        def fDP(fpr):
            fnr = 0.0
            return fnr

        self.RenyiDP = RenyiDP
        self.approxRDP = approxRDP
        self.approxDP = approxDP
        self.fDP = fDP

        self.eps_pureDP = np.inf  # equivalent to RenyiDP(np.inf) and approxDP(0).

        self.delta0 = np.inf  # indicate the smallest allowable \delta0 in approxRDP that is not inf

        self.group_size = 1  # transformation that increases group size.
        self.replace_one = False  #
        self.local_flag = False  # for the purpose of implementating local DP.
        #  We can convert localDP to curator DP by parallel composition and by shuffling.
        self.updated = False  # if updated, then when getting eps, we know that directly getting
        # approxDP is the tightest possible.

    def get_approxDP(self, delta):
        # Output eps as a function of delta
        return self.approxDP(delta)

    def get_approxRDP(self, delta, alpha):
        # Output eps as a function of delta and alpha
        return self.approxRDP(delta, alpha)

    def get_RDP(self, alpha):
        # Output RDP eps as a function of alpha
        return self.RenyiDP(alpha)

    def get_fDP(self, fpr):
        # Output false negative rate as a function of false positive rate
        return self.fDP(fpr)

    def get_pureDP(self):
        return self.eps_pureDP

    def get_eps(self, delta):
        # Get the smallest eps fo multiple calculations
        eps = [self.get_pureDP(), self.get_approxDP(delta)]
        # add get eps from RDP and get eps from approx RDP
        # and check the 'updated' flag. if updated, no need to do much
        return np.min(eps)

    def propagate_updates(self, func, type_of_update,
                          delta0=0,
                          BBGHS_conversion=True,
                          fDP_based_conversion=False):
        # This function receives a new description of the mechanisms and updates all functions
        # based on what is new by calling converters.

        if type_of_update == 'pureDP':
            # function is one number
            eps = func
            self.eps_pureDP = np.minimum(eps, self.eps_pureDP)

            approxdp_new = converter.puredp_to_approxdp(eps)
            self.approxDP = converter.pointwise_minimum(approxdp_new, self.approxDP)
            rdp_new = converter.puredp_to_rdp(eps)
            self.RenyiDP = converter.pointwise_minimum(rdp_new, self.RenyiDP)
            fdp_new = converter.puredp_to_fdp(eps)
            self.fDP = converter.pointwise_maximum(fdp_new, self.fDP)

            self.approxRDP = converter.approxdp_func_to_approxrdp(self.approxDP)

            self.delta0 = 0  # the minimum non-trivial approximate RDP is now 0

            # lambda x: np.maximum(fdp_new(x),self.fDP(x))
        elif type_of_update == 'approxDP':
            # func will be a tuple of two numbers
            eps = func[0]
            delta = func[1]

            self.approxRDP = converter.pointwise_minimum_two_arguments(self.approxRDP,
                                          converter.approxdp_to_approxrdp(eps, delta))

            def approx_dp_func(delta1):
                if delta1 >= delta:
                    return eps
                else:
                    return np.inf

            self.approxDP = converter.pointwise_minimum(self.approxDP, approx_dp_func)
            self.fDP = converter.pointwise_maximum(self.fDP, converter.approxdp_to_fdp(eps, delta))

            self.delta0 = np.minimum(self.delta0, delta)
        elif type_of_update == 'approxDP_func':
            # func outputs eps as a function of delta
            # optional input delta0, telling us from where \epsilon becomes infinity

            self.delta0 = np.minimum(delta0, self.delta0)
            self.fDP = converter.pointwise_maximum(self.fDP, converter.approxdp_func_to_fdp(func))
            self.approxRDP = converter.pointwise_minimum_two_arguments(self.approxRDP,
                                 converter.approxdp_func_to_approxrdp(func))
            self.approxDP = converter.pointwise_minimum(self.approxDP, func)

        elif type_of_update == 'RDP':
            # function output RDP eps as a function of alpha
            self.RenyiDP = converter.pointwise_minimum(self.RenyiDP, func)

            if fDP_based_conversion:

                fdp_log, fdp_grad_log = converter.rdp_to_fdp_and_fdp_grad_log(func)

                self.fDP = converter.pointwise_maximum(self.fDP, converter.rdp_to_fdp(self.RenyiDP))

                # # --------- debugging code below -----------------
                #
                # def fdp_grad(x):
                #     return -np.exp(fdp_grad_log(np.log(x)))[0]
                #
                # def plot_fdp(x):
                #     grad = fdp_grad(x)
                #     y = self.fDP(x)
                #
                #     def tangent_line(u):
                #         return y + grad*(u-x)
                #
                #     import matplotlib.pyplot as plt
                #
                #     fpr_list, fnr_list = self.plot_fDP()
                #     plt.figure(1)
                #     plt.plot(fpr_list, fnr_list)
                #     plt.plot(fpr_list, tangent_line(fpr_list))
                #     plt.show()
                #
                # plot_fdp(0.01)
                # # ------------------------------------------------

                self.approxDP = converter.pointwise_minimum(self.approxDP,
                                                            converter.fdp_fdp_grad_to_approxdp(
                                                                fdp_log, fdp_grad_log,
                                                                log_flag=True))

                # self.approxDP = converter.pointwise_minimum(self.approxDP,
                #                                            converter.fdp_to_approxdp(self.fDP))
            else:
                self.approxDP = converter.pointwise_minimum(self.approxDP,
                         converter.rdp_to_approxdp(self.RenyiDP, BBGHS_conversion=BBGHS_conversion))
                self.fDP = converter.pointwise_maximum(self.fDP,
                                                       converter.approxdp_func_to_fdp(
                                                           self.approxDP))


        elif type_of_update == 'fDP':
            # f-DP,  input is Type I error or fpr, output is Type II error or fnr
            self.fDP = converter.pointwise_maximum(self.fDP, func)

            self.approxDP = converter.pointwise_minimum(self.approxDP,
                                                        converter.fdp_to_approxdp(func))
            self.approxRDP = converter.pointwise_minimum(self.approxRDP,
                                                         converter.approxdp_func_to_approxrdp(
                                                             self.approxDP))
        elif type_of_update == 'fDP_and_grad':
            # the input will be two functions
            fdp = func[0]
            fdp_grad = func[1]
            self.fdp = converter.pointwise_maximum(self.fDP, fdp)
            self.approxDP = converter.pointwise_minimum(self.approxDP,
                                                        converter.fdp_fdp_grad_to_approxdp(fdp,
                                                              fdp_grad,log_flag=False))
            self.approxRDP = converter.pointwise_minimum(self.approxRDP,
                                                         converter.approxdp_func_to_approxrdp(
                                                             self.approxDP))
        elif type_of_update == 'fDP_and_grad_log':
            # the input will be two functions
            fun1 = func[0]
            fun2 = func[1]
            fdp = lambda x: 1 - np.exp(fun1(np.log(x)))
            self.fDP = converter.pointwise_maximum(self.fDP, fdp)
            self.approxDP = converter.pointwise_minimum(self.approxDP,
                                                        converter.fdp_fdp_grad_to_approxdp(fun1,
                                                                                           fun2,
                                                                                           log_flag=True))
            self.approxRDP = converter.pointwise_minimum(self.approxRDP,
                                                         converter.approxdp_func_to_approxrdp(
                                                             self.approxDP))
        elif type_of_update == 'approxRDP':
            # func is a function of alpha and delta
            self.delta0 = np.minimum(delta0, self.delta0)
            self.approxRDP = converter.pointwise_minimum_two_arguments(self.approxRDP, func)
            # TODO: Write a function that converts approximateRDP to approximateDP

            # TODO: Write a function that converts approximateRDP to fDP.
        else:
            print(type_of_update, ' not recognized.')


    def set_all_representation(self, mech):
        """
        Set all representations from a mechanism object. This is useful when constructing a
        mechanism object using transformers then wanted to convert to a mechanism class definition.

        :param mech:  Input mechanism object.
        :return: None
        """
        # Need to make sure that the input is a mechanism object

        self.approxDP = mech.approxDP
        self.RenyiDP = mech.RenyiDP
        self.fDP = mech.fDP
        self.eps_pureDP = mech.eps_pureDP
        self.delta0 = mech.delta0

    # Plotting functions: returns lines for people to plot outside

    # Plot FNR against FPR --- hypothesis testing interpretation of DP
    def plot_fDP(self, length=101):
        fpr_list = np.linspace(0, 1, length)
        fnr_list = np.array([self.get_fDP(fpr) for fpr in fpr_list])
        return fpr_list, fnr_list

    # Plot RDP function
    def plot_RDP(self, alphamax=101, length=101):
        alpha_list = np.linspace(0, alphamax, length)
        RDP_list = np.array([self.get_RDP(alpha) for alpha in alpha_list])
        return alpha_list, RDP_list


class Transformer():
    """
    A transformer is a callable object that takes one or more mechanism as input and
    **transform** them into a new mechanism
    """

    def __init__(self):
        self.name = 'generic_transformer'
        self.unary_operator = False  # If true it takes one mechanism as an input,
        # otherwise it could take many, e.g., composition
        self.preprocessing = False  # Relevant if unary is true, it specifies whether the operation
        # is before or after the mechanism, e.g., amplification by sampling is before applying the
        # mechanism, "amplification by shuffling" is after applying the LDP mechanisms
        self.transform = lambda x: x

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class Calibrator():
    """
    A calibrator calibrates noise (or other parameters) meet a pre-scribed privacy budget
    """

    def __init__(self):
        self.name = 'generic_calibrator'

        self.eps_budget = np.inf
        self.delta_budget = 1.0

        self.obj_func = lambda x: 0

        self.calibrate = lambda x: x
        # Input privacy budget, a mechanism with params,  output a set of params that works
        # while minimizing the obj_func as much as possible

    def __call__(self, *args, **kwargs):
        return self.calibrate(*args, **kwargs)
