"""
Example of a specific transformer that outputs the composition
1. class Composition: supports RDP-based composition.
2. class ComposeAFA: supports characteristic function (phi-function) based composition.
3. class ComposeGaussian: supports composition of only Gaussian mechanisms.
4. class AmplificationBySampling: supports RDP and DP based amplification rule.
5. classAmplificationBySampling_pld: amplification by sampling for privacy loss distribution.

"""

from autodp.autodp_core import Mechanism, Transformer
from copy import deepcopy
import math
import numpy as np

from autodp import mechanism_zoo, rdp_acct, phi_bank


# The generic composition class
class Composition(Transformer):
    """ Composition is a transformer that takes a list of Mechanisms and number of times they appear,
    and output a Mechanism that represents the composed mechanism"""
    def __init__(self):
        Transformer.__init__(self)
        """
        Args:

        """
        self.name = 'Composition'

        # Update the function that is callable
        self.transform = self.compose

    def compose(self, mechanism_list, coeff_list, RDP_compose_only=True, BBGHS_conversion=True, fDP_based_conversion=False):
        # Make sure that the mechanism has a unique list
        # for example, if there are two Gaussian mechanism with two different sigmas, call it
        # Gaussian1, and Gaussian2

        # if RDP_compose_only is true, we use only RDP composition.


        newmech = Mechanism()

        # update the functions
        def newrdp(x):
            return sum([c * mech.RenyiDP(x) for (mech, c) in zip(mechanism_list, coeff_list)])
        newmech.propagate_updates(newrdp, 'RDP',  BBGHS_conversion= BBGHS_conversion, fDP_based_conversion=fDP_based_conversion)

        # TODO: the fDP_based_conversion sometimes fails due to undefined RDP with alpha < 1

        newmech.eps_pureDP = sum([c * mech.eps_pureDP for (mech, c)
                                  in zip(mechanism_list, coeff_list)])
        newmech.delta0 = max([mech.delta0 for (mech, c)
                              in zip(mechanism_list, coeff_list)])

        if not RDP_compose_only:  # Also do KOV-composition while optimizing over \delta parameter
            # TODO: Also implement the KOV-composition here and propagate the updates
            # TODO: How do we generically compose eps(delta) functions?
            # TODO: How do we generically compose approximate RDP functions
            # TODO: How do we generically compose fDP? (efficiently)
            pass

        # Other book keeping
        newmech.name = self.update_name(mechanism_list, coeff_list)
        # keep track of all parameters of the composed mechanisms
        newmech.params = self.update_params(mechanism_list)

        return newmech

    def update_name(self,mechanism_list, coeff_list):
        separator = ', '
        s = separator.join([mech.name + ': ' + str(c) for (mech, c)
                           in zip(mechanism_list, coeff_list)])

        return 'Compose:{'+ s +'}'

    def update_params(self, mechanism_list):
        params = {}
        for mech in mechanism_list:
            params_cur = {mech.name+':'+k: v for k,v in mech.params.items()}
            params.update(params_cur)
        return params



# The generic composition class
class ComposeAFA(Transformer):
    """ The analytical Fourier Accountant (AFA) is a transformer that takes a list of Mechanisms and number of times they appear,
    and output a Mechanism that represents the composed mechanism.
    https://arxiv.org/pdf/2106.08567.pdf
    """
    def __init__(self):
        Transformer.__init__(self)
        self.name = 'ComposeFourier'

        # Update the function that is callable
        self.transform = self.compose

    def compose(self, mechanism_list, coeff_list):
        """
        In the composition, we keep track of two lists of characteristic functions (Phi(t) and Phi'(t))with
        respect to the privacy loss R.V. log(p/q) and log(q/p).
        For most basic mechanisms (e.g., Gaussian mechanism, Lapalce mechanism), their phi(t) and Phi'(t) are the same.
        For some advanced mechanisms (e.g., SubsampleGaussian mechanism), their characteristic functions are not symmetric.
        """

        newmech = Mechanism()

        # update the functions: log_phi_p and log_phi_q
        def new_log_phi_p2q(x):
            return sum([c * mech.log_phi_p2q(x) for (mech, c) in zip(mechanism_list, coeff_list)])

        def new_log_phi_q2p(x):
            return sum([c * mech.log_phi_q2p(x) for (mech, c) in zip(mechanism_list, coeff_list)])



        newmech.propagate_updates((new_log_phi_p2q, new_log_phi_q2p), 'log_phi')
        # Other book keeping
        newmech.name = self.update_name(mechanism_list, coeff_list)
        # keep track of all parameters of the composed mechanisms
        newmech.params = self.update_params(mechanism_list)

        return newmech

    def update_name(self,mechanism_list, coeff_list):
        separator = ', '
        s = separator.join([mech.name + ': ' + str(c) for (mech, c)
                           in zip(mechanism_list, coeff_list)])

        return 'Compose:{'+ s +'}'

    def update_params(self, mechanism_list):
        params = {}
        for mech in mechanism_list:
            params_cur = {mech.name+':'+k: v for k,v in mech.params.items()}
            params.update(params_cur)
        return params




# composition of only Gaussian mechanisms
class ComposeGaussian(Composition):
    """ CompositionGaussian is a specialized composation function of ONLY Guassian mechanisms
    output a Mechanism that represents the composed mechanism"""
    def __init__(self):
        Composition.__init__(self)
        self.name = 'ComposeGaussian'

    def compose(self, mechanism_list, coeff_list):
        # Make sure that the list contains only Gaussian mechanisms
        for mech in mechanism_list:
            assert(isinstance(mech, mechanism_zoo.GaussianMechanism)
                   or isinstance(mech, mechanism_zoo.ExactGaussianMechanism))

        # Directly compose the distribution of privacy-loss random variables.
        # Sum of Gaussians are gaussians
        tmp = 0
        for mech, coeff in zip(mechanism_list,coeff_list):
            tmp += mech.params['sigma']**(-2)*coeff

        newmech = mechanism_zoo.ExactGaussianMechanism(sigma=math.sqrt(1/tmp))

        # Other book keeping
        newmech.name = self.update_name(mechanism_list, coeff_list)

        # keep track of all parameters of the composed mechanisms
        newmech.params = self.update_params(mechanism_list)

        return newmech





class AmplificationBySampling_pld(Transformer):
    def __init__(self, PoissonSampling=True, neighboring='add_only'):
        # Amplification By Sampling rule for PLD formalism
        # By default, poisson sampling is used:  sample a dataset by selecting each data point iid
        # If PoissonSampling is set to False, then it chooses a random subset with size prob * n
        # Compared to other amplification rules (e.g., rules in RDP), this rule treats neighboring relationship separately:
        # this amplification rule is for either "add_only" or "remove_only" relationship.
        Transformer.__init__(self)
        if PoissonSampling:
            self.name = 'PoissonSample'
        else:
            self.name = 'Subsample'
        self.PoissonSampling = PoissonSampling
        self.unary_operator = True
        self.preprocessing = True # Sampling happen before the mechanism is applied
        self.neighboring = neighboring

        # Update the function that is callable
        self.transform = self.amplify

    def amplify(self, mechanism, prob):
        # If the neighboring relationship is adding, the dominating distribution will be (P, (1-gamma)P + gamma*Q)
        # If the neighboring relationship is removing, the dominating distribution will be ( (1-gamma)*P + gamma*Q, P)

        newmech = Mechanism()

        # privacy amplification via approx-dp

        # Amplification of RDP
        # propagate to approxDP as well.
        # amplification by sampling for adding relationship is compatible with the base mechanism that has adding or
        # add/remove or replace relationship.
        if self.neighboring == 'add_only':
            assert  mechanism.neighboring != 'remove_only', "mechanism's neighboring relationship notion of DP is " \
                                                   "incompatible with Privacy Amplification by  sampling"
            # take care of the add neighbor and remove neighbor
            # If not, there actually isn't a way to convert it from replace-one notation,
            # unless a "dummy" user exists in the space.
            newmech.neighboring = 'add_only'
            # new_pdf_p and new_pdf_q denote the new dominating pair of sampled mechanism.
            new_pdf_p = lambda x: mechanism.pdf_p(x)
            new_pdf_q = lambda x: (1. - prob) * mechanism.pdf_p(x) + prob * mechanism.pdf_q(x)
            newmech.propagate_updates((new_pdf_p, new_pdf_q),'pdf')
            # Besides using general pdf2phi conversion, we can construct a more numerical stable phi function.
            # In the subsampled Gaussian case, the following phi functions is more stable as many exponential terms will
            # cancel out in the phi function.
            # Check if the base mechanism is Gaussian mechanism. If so, use available more-stable phi functions.
            if len(mechanism.params.keys()) == 1 and 'sigma' in mechanism.params.keys():
                params = {'sigma': mechanism.params['sigma'], 'gamma':prob}
                log_phi_p = lambda x: phi_bank.phi_subsample_gaussian_p(params, x, remove_only=False)
                log_phi_q = lambda x: phi_bank.phi_subsample_gaussian_q(params, x, remove_only = False)
                newmech.propagate_updates((log_phi_p, log_phi_q), 'log_phi')

        else:
            assert mechanism.neighboring != 'add_only', "mechanism's neighboring relationship notion of DP is " \
                                                   "incompatible with Privacy Amplification "
            newmech.neighboring ='remove_only'
            if self.PoissonSampling:
                new_pdf_p = lambda x: (1. - prob) * mechanism.pdf_q(x) + prob * mechanism.pdf_p(x)
                new_pdf_q = lambda x: mechanism.pdf_q(x)
                newmech.propagate_updates((new_pdf_p, new_pdf_q), 'pdf')
                # Check if the base mechanism is Gaussian mechanism. If so, use available more-stable phi functions.
                if len(mechanism.params.keys()) == 1 and 'sigma'in mechanism.params.keys():
                    params = {'sigma': mechanism.params['sigma'], 'gamma': prob}
                    log_phi_p2q = lambda x: phi_bank.phi_subsample_gaussian_p(params, x, remove_only=True)
                    log_phi_q2p = lambda x: phi_bank.phi_subsample_gaussian_q(params, x, remove_only=True)
                    newmech.propagate_updates((log_phi_p2q, log_phi_q2p), 'log_phi')
            else:
                # subset sampling
                new_pdf_p = lambda x: (1. - prob) * mechanism.pdf_p(x) + prob * mechanism.pdf_q(x)
                new_pdf_q = lambda x: mechanism.pdf_p(x)
        # convert pdf to phi
        newmech.propagate_updates([new_pdf_p, new_pdf_q], 'pdf')

        if prob == 0:
            new_approxDP = lambda delta:0
        else:
            new_approxDP = lambda delta: np.log(1 + prob*(np.exp(mechanism.approxDP(delta/prob))-1))
        newmech.approxDP = new_approxDP


        # TODO: For future work, we need to update subsampled CDF


        #TODO: Implement the amplification of f-DP
        # propagate to approxDP, or simply get the f-DP from approximate-DP.


        # book keeping
        key = self.name + '_' + str(prob)
        num = 0
        newname = self.name

        # the following handles the case when key is already in the params
        while key in mechanism.params:
            num = num+1
            newname = self.name+str(num)
            key = newname + '_' + str(prob)

        newmech.name = newname +':'+mechanism.name
        newmech.params = deepcopy(mechanism.params)
        new_params = {newname:prob}
        newmech.params.update(new_params)

        return newmech



class AmplificationBySampling(Transformer):
    def __init__(self, PoissonSampling=True):
        # By default, poisson sampling is used:  sample a dataset by selecting each data point iid
        # If PoissonSampling is set to False, then it chooses a random subset with size prob * n
        Transformer.__init__(self)
        if PoissonSampling:
            self.name = 'PoissonSample'
        else:
            self.name = 'Subsample'
        self.PoissonSampling = PoissonSampling
        self.unary_operator = True
        self.preprocessing = True # Sampling happen before the mechanism is applied

        # Update the function that is callable
        self.transform = self.amplify

    def amplify(self, mechanism, prob, improved_bound_flag=False):
        # If you know that your mechanism
        #  - (for PoissonSampling) satisfies the the conditions in Theorem 8 of http://proceedings.mlr.press/v97/zhu19c/zhu19c.pdf
        #  - or (for subsampling)  satisfies the conditions of Theorem 27 of https://arxiv.org/pdf/1808.00087.pdf
        # then you may switch general_bound_flag to False to get a tighter bound.

        # Else, for all mechanisms with RDP bounds, the general upper bounds are used by default.

        newmech = Mechanism()

        # privacy amplification via approx-dp



        # Amplification of RDP
        # propagate to approxDP as well.

        if self.PoissonSampling:
            assert mechanism.neighboring is not 'replace_one', "mechanism's replace_one notion of DP is " \
                                                   "incompatible with Privacy Amplification " \
                                                   "by Poisson sampling"
            # check that the input mechanism uses the standard add-or-remove notion of DP.
            # If not, there actually isn't a way to convert it from replace-one notation,
            # unless a "dummy" user exists in the space.
            newmech.replace_one = False

        else:  # if we want subset subsampled DP
            assert mechanism.neighboring is not 'add_remove', "mechanism's add-remove notion of DP is " \
                                                   "incompatible with Privacy Amplification " \
                                                   "by subsampling without replacements"
            # TODO: implement a transformer that convert add/remove to replace_one notion of DP.
            newmech.replace_one = True

        if prob == 0:
            new_approxDP = lambda delta:0
        else:
            new_approxDP = lambda delta: np.log(1 + prob*(np.exp(mechanism.approxDP(delta/prob))-1))
        newmech.approxDP = new_approxDP

        acct = rdp_acct.anaRDPacct()
        if self.PoissonSampling:
            if improved_bound_flag:
                acct.compose_poisson_subsampled_mechanisms(mechanism.RenyiDP,prob)
            else:
                acct.compose_poisson_subsampled_mechanisms1(mechanism.RenyiDP,prob)
        else:  # subsampling
            if improved_bound_flag:
                acct.compose_subsampled_mechanism(mechanism.RenyiDP, prob, improved_bound_flag=True)
            else:
                acct.compose_subsampled_mechanism(mechanism.RenyiDP, prob)


        acct.build_zeroth_oracle()
        new_rdp = acct.evalRDP
        newmech.propagate_updates(new_rdp,'RDP')

        #TODO: Implement the amplification of f-DP
        # propagate to approxDP, or simply get the f-DP from approximate-DP.


        # book keeping
        key = self.name + '_' + str(prob)
        num = 0
        newname = self.name

        # the following handles the case when key is already in the params
        while key in mechanism.params:
            num = num+1
            newname = self.name+str(num)
            key = newname + '_' + str(prob)

        newmech.name = newname +':'+mechanism.name
        newmech.params = deepcopy(mechanism.params)
        new_params = {newname:prob}
        newmech.params.update(new_params)

        return newmech




# TODO: implement other transformers:
# - amplification by shuffling
# - parallel composition
# - group composition
# - private selection of private candidates
# - amplification by overwhelmingly large-probability event.