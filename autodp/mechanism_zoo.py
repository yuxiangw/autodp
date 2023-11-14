"""
'mechanism_zoo' module implements popular DP mechanisms with their privacy guarantees

"""
import math
from scipy import special
import numpy as np
from autodp.autodp_core import Mechanism
from autodp import rdp_bank, dp_bank, fdp_bank, phi_bank
from autodp import transformer_zoo
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import numpy as np


# Example of a specific mechanism that inherits the Mechanism class
class GaussianMechanism(Mechanism):
    """
    The example of Gaussian mechanism with different characterizations.
    """
    def __init__(self, sigma, name='Gaussian',
                 RDP_off=False, approxDP_off=False, fdp_off=True,
                 use_basic_RDP_to_approxDP_conversion=False,
                 use_fDP_based_RDP_to_approxDP_conversion=False, phi_off=True):
        """
        sigma: the std of the noise divide by the l2 sensitivity.
        coeff: the number of composition
        RDP_off: if False, then we characterize the mechanism using RDP.
        fdp_off: if False, then we characterize the mechanism using fdp.
        phi_off: if False, then we characterize the mechanism using phi-function.
        """
        Mechanism.__init__(self)

        self.name = name # When composing
        self.params = {'sigma': sigma} # This will be useful for the Calibrator
        # TODO: should a generic unspecified mechanism have a name and a param dictionary?

        self.delta0 = 0

        if not phi_off:
            """
            Apply phi function to analyze Gaussian mechanism.
            the CDF of privacy loss R.V. is computed using an integration (see details in cdf_bank) through Levy Theorem.
            """
            log_phi = lambda x: phi_bank.phi_gaussian({'sigma': sigma}, x)
            self.exact_phi = True
            # self.cdf tracks the cdf of log(p/q) and the cdf of log(q/p).
            self.propagate_updates((log_phi, log_phi), 'log_phi')

            # Propagate the pdf of dominating pairs.
            def pdf_p(x): return norm.pdf((x-1.), scale=sigma**2)
            def pdf_q(x): return norm.pdf(x, scale=sigma**2)
            self.propagate_updates((pdf_p, pdf_q), 'pdf')
            """
            Moreover, we know the closed-form expression of the CDF of the privacy loss RV
               privacy loss RV distribution l=log(p/q) ~ N(1/2\sigma^2, 1/sigma^2)
            We can also use the following closed-form cdf directly.
            """
            #sigma = sigma*1.0/np.sqrt(coeff)
            #mean = 1.0 / (2.0 * sigma ** 2)
            #std = 1.0 / (sigma)
            #cdf = lambda x: norm.cdf((x - mean) / std)
            #self.propagate_updates(cdf, 'cdf', take_log=True)


        if not RDP_off:
            new_rdp = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
            if use_fDP_based_RDP_to_approxDP_conversion:
                # This setting is slightly more complex, which involves converting RDP to fDP,
                # then to eps-delta-DP via the duality
                self.propagate_updates(new_rdp, 'RDP', fDP_based_conversion=True)
            elif use_basic_RDP_to_approxDP_conversion:
                self.propagate_updates(new_rdp, 'RDP', BBGHS_conversion=False)
            else:
                # This is the default setting with fast computation of RDP to approx-DP
                self.propagate_updates(new_rdp, 'RDP')

        if not approxDP_off: # Direct implementation of approxDP
            new_approxdp = lambda x: dp_bank.get_eps_ana_gaussian(sigma, x)
            self.propagate_updates(new_approxdp,'approxDP_func')

        if not fdp_off: # Direct implementation of fDP
            fun1 = lambda x: fdp_bank.log_one_minus_fdp_gaussian({'sigma': sigma}, x)
            fun2 = lambda x: fdp_bank.log_neg_fdp_grad_gaussian({'sigma': sigma}, x)
            self.propagate_updates([fun1,fun2],'fDP_and_grad_log')
            # overwrite the fdp computation with the direct computation
            self.fdp = lambda x: fdp_bank.fDP_gaussian({'sigma': sigma}, x)

        # the fDP of gaussian mechanism is equivalent to analytical calibration of approxdp,
        # so it should have been automatically handled numerically above


        # Discussion:  Sometimes delta as a function of eps has a closed-form solution
        # while eps as a function of delta does not
        # Shall we represent delta as a function of eps instead?


class ExactGaussianMechanism(Mechanism):
    """
    The Gaussian mechanism to use in practice with tight direct computation of everything
    """
    def __init__(self, sigma=None, name='Gaussian'):
        # the sigma parameter is the std of the noise divide by the l2 sensitivity
        Mechanism.__init__(self)

        self.name = name # When composing
        self.params = {'sigma': sigma} # This will be useful for the Calibrator
        self.delta0 = 0
        if sigma is not None:
            new_rdp = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
            self.propagate_updates(new_rdp, 'RDP')
            # Overwrite the approxDP and fDP with their direct computation
            self.approxDP = lambda x: dp_bank.get_eps_ana_gaussian(sigma, x)
            self.fDP = lambda x: fdp_bank.fDP_gaussian({'sigma': sigma}, x)


class LaplaceMechanism(Mechanism):
    """
    The Laplace Mechanism that support RDP and phi-function based characterization.
    """
    def __init__(self, b=None, name='Laplace', RDP_off=False, phi_off=True):
        """
        b: the ratio of the scale parameter and L1 sensitivity.
        RDP_off: if False, then we characterize the mechanism using RDP.
        fdp_off: if False, then we characterize the mechanism using fdp.
        phi_off: if False, then we characterize the mechanism using phi-function.
        """
        Mechanism.__init__(self)

        self.name = name
        self.params = {'b': b} # This will be useful for the Calibrator

        self.delta0 = 0
        if not phi_off:

            log_phi_p2q = lambda x: phi_bank.phi_laplace(self.params, x)
            log_phi_q2p = lambda x: phi_bank.phi_laplace(self.params, x)

            self.log_phi_p2q = log_phi_p2q
            self.log_phi_q2p = log_phi_q2p
            self.propagate_updates((log_phi_p2q, log_phi_q2p), 'log_phi')


        if not RDP_off:
            new_rdp = lambda x: rdp_bank.RDP_laplace({'b': b}, x)
            self.propagate_updates(new_rdp, 'RDP')



class RandresponseMechanism(Mechanism):

    """
    The randomized response mechanism that supports RDP and phi-function based characterization.
    TODO: assert when p is None.
    """
    def __init__(self, p=None, RDP_off=False, phi_off=True, name='Randresponse'):
        """
        p: the Bernoulli probability p of outputting the truth.
        """
        Mechanism.__init__(self)

        self.name = name
        self.params = {'p': p}
        self.delta0 = 0

        if not RDP_off:
            new_rdp = lambda x: rdp_bank.RDP_randresponse({'p': p}, x)
            self.propagate_updates(new_rdp, 'RDP')
            approxDP = lambda x: dp_bank.get_eps_randresp_optimal(p, x)
            self.propagate_updates(approxDP, 'approxDP_func')

        if not phi_off:
            log_phi = lambda x: phi_bank.phi_rr_p({'p': p, 'q':1-p}, x)
            self.propagate_updates((log_phi, log_phi), 'log_phi')

class zCDP_Mechanism(Mechanism):
    def __init__(self,rho,xi=0,name='zCDP_mech'):
        Mechanism.__init__(self)

        self.name = name
        self.params = {'rho':rho,'xi':xi}
        new_rdp = lambda x: rdp_bank.RDP_zCDP(self.params, x)

        self.propagate_updates(new_rdp,'RDP')


class ExponentialMechanism(zCDP_Mechanism):
    def __init__(self, eps, RDP_off=False, phi_off=True, non_adaptive=False, name='ExpMech'):
        zCDP_Mechanism.__init__(self, eps**2/8, name=name)
        # the zCDP bound is from here: https://arxiv.org/pdf/2004.07223.pdf

        self.eps_pureDP = eps
        self.propagate_updates(eps, 'pureDP')

        if not RDP_off:
            new_rdp = lambda x: rdp_bank.RDP_pureDP({'eps': eps}, x)
            self.propagate_updates(new_rdp, 'RDP')

        def func(t):
            """
            Return the bernoulli parameter p and q for any t.

            """
            p = (np.exp(-t) - np.exp(-eps))/(1-np.exp(-eps))
            q = (1 - np.exp(t-eps))/(1-np.exp(-eps))
            return {'p':p, 'q':q}
        # TODO: implement the f-function and phi-function representation from two logistic r.v.


        if not phi_off:
            # t is from [0, eps], log(p/q) in [t-eps, t], we optimize t over compositions.
            #  see page 10 for the dominating pair distribution. https://arxiv.org/pdf/1909.13830.pdf
            log_phi_p2q = lambda x, t: phi_bank.phi_rr_p(func(t), x)
            log_phi_q2p = lambda x, t: phi_bank.phi_rr_q(func(t), x)
            self.tbd_range = [0, eps]
            self.propagate_updates((log_phi_p2q, log_phi_q2p), 'log_phi_adv')




class PureDP_Mechanism(Mechanism):
    def __init__(self, eps, name='PureDP'):
        # the eps parameter is the pure DP parameter of this mechanism
        Mechanism.__init__(self)

        self.name = name # Used for generating new names when composing
        self.params = {'eps': eps} #

        self.propagate_updates(eps, 'pureDP')

        # ------- I verified that the following options give the same results ----
        # def new_rdp(x):
        #     return rdp_bank.RDP_pureDP({'eps': eps}, x)
        #
        # if use_basic_RDP_to_approxDP_conversion:
        #     self.propagate_updates(new_rdp, 'RDP', BBGHS_conversion=False)
        # else:


        #     self.propagate_updates(new_rdp, 'RDP')

class zCDP_Mechanism(Mechanism):
    def __init__(self,rho,xi=0,name='zCDP_mech'):
        Mechanism.__init__(self)

        self.name = name
        self.params = {'rho':rho,'xi':xi}
        new_rdp = lambda x: rdp_bank.RDP_zCDP(self.params, x)

        self.propagate_updates(new_rdp,'RDP')


class DiscreteGaussianMechanism(zCDP_Mechanism):
    def __init__(self, sigma, name='DGM'):
        zCDP_Mechanism.__init__(self, 0.5/sigma**2, name=name)

        # This the the best implementation for DGM for now.
        # The analytical formula for approximate-DP applies only to 1D outputs
        # The exact eps,delta-DP and char function for the multivariate outputs
        # requires a further maximization over neighboring datasets, which is unclear how to do

class ExponentialMechanism(zCDP_Mechanism):
    def __init__(self, eps, name='ExpMech'):
        zCDP_Mechanism.__init__(self, eps**2/8, name=name)
        # the zCDP bound is from here: https://arxiv.org/pdf/2004.07223.pdf
        self.params['eps'] = eps

        # TODO: Bounded range should imply a slightly stronger RDP that dominates the following
        self.eps_pureDP = eps
        self.propagate_updates(eps, 'pureDP')

        # TODO: implement the f-function and phi-function representation from two logistic r.v.



class SubsampleGaussianMechanism(Mechanism):
    """
    Poisson subsampled Gaussian mechanism.

    unlike the general mechanism design, we specify the number of composition as an input to initialize the mechanism.
    Therefore, we can use this mechanism as an example for privacy calibration: calibrate the noise scale for a subsample
    gaussian mechanism that runs for 'coeff' rounds.

    """
    def __init__(self,params, phi_off=True, RDP_off=False, neighboring='remove_only', name='SubsampleGaussian'):
        Mechanism.__init__(self)
        self.name=name
        self.params={'prob':params['prob'],'sigma':params['sigma'],'coeff':params['coeff']}

        if not RDP_off:
            # create such a mechanism as in previously
            subsample = transformer_zoo.AmplificationBySampling()  # by default this is using poisson sampling
            mech = GaussianMechanism(sigma=params['sigma'])

            # Create subsampled Gaussian mechanism
            SubsampledGaussian_mech = subsample(mech, params['prob'], improved_bound_flag=True)

            # Now run this for niter iterations
            compose = transformer_zoo.Composition()
            mech = compose([SubsampledGaussian_mech], [params['coeff']])

            # Now we get it and let's extract the RDP function and assign it to the current mech being constructed
            rdp_total = mech.RenyiDP
            self.propagate_updates(rdp_total, type_of_update='RDP')

        if not phi_off:
            """
            # phi-function based characterization takes care of add/remove neighboring relationship separately, see
            # https://arxiv.org/pdf/2106.08567.pdf
            # We can obtain the results for the standard add/remove by a pointwise maximum of the two.
            # For example, we define mech_add and mech_remove as follow:
            mech_add = SubsampleGaussianMechanism({'prob':prob, 'sigma':sigma, 'coeff': coeff}, phi_off=False,
             neighboring='add_only')
            mech_remove = SubsampleGaussianMechanism({'prob':prob, 'sigma':sigma, 'coeff': coeff}, phi_off=False,
             neighboring='remove_only')
            # Query epsilon at delta for the standard add/remove:
            eps = max(mech_add.approxDP(delta), mch_remove.approxDP(delta))
            
            """
            n_comp = params['coeff']
            if neighboring == 'remove_only':
                # log_phi_p2q denotes the approximated phi-function of the privacy loss R.V. log(p/q).
                # log_phi_q2p denotes the approximated phi-function of the privacy loss R.V. log(q/p).
                composed_log_phi_p2q = lambda x: n_comp * phi_bank.phi_subsample_gaussian_p\
                    (self.params, x, remove_only=True)
                composed_log_phi_q2p = lambda x: n_comp * phi_bank.phi_subsample_gaussian_q\
                    (self.params, x, remove_only=True)
                self.neighboring = 'remove_only'

            else:
                # neighboring for add_only
                composed_log_phi_p2q = lambda x: n_comp * phi_bank.phi_subsample_gaussian_p\
                    (self.params, x, remove_only=False)
                composed_log_phi_q2p = lambda x: n_comp * phi_bank.phi_subsample_gaussian_q\
                    (self.params, x, remove_only=False)
                self.neighboring = 'add_only'
            self.propagate_updates((composed_log_phi_p2q, composed_log_phi_q2p), 'log_phi')


class ComposedGaussianMechanism(Mechanism):
    """
    This one is used as an example for calibrator with composed Gaussian mechanism
    """
    def __init__(self,params,name='SubsampleGaussian'):
        Mechanism.__init__(self)
        self.name=name
        self.params={'sigma':params['sigma'],'coeff':params['coeff']}
        # create such a mechanism as in previously

        mech = GaussianMechanism(sigma=params['sigma'])
        # Now run this for coeff iterations
        compose = transformer_zoo.Composition()
        mech = compose([mech], [params['coeff']])

        # Now we get it and let's extract the RDP function and assign it to the current mech being constructed
        rdp_total = mech.RenyiDP
        self.propagate_updates(rdp_total, type_of_update='RDP')



class NoisyScreenMechanism(Mechanism):
    """
    The data-dependent RDP of ``Noisy Screening" (Theorem 7 in Private-kNN (CVPR-20))
    This mechanism is also used in Figure 2(a) in Gaussian-based SVTs.
    https://papers.nips.cc/paper/2020/file/e9bf14a419d77534105016f5ec122d62-Paper.pdf
    """
    def __init__(self,params,name='NoisyScreen'):
        Mechanism.__init__(self)
        self.name=name
        self.params={'logp':params['logp'],'logq':params['logq']}
        # create such a mechanism as in previously

        new_rdp = lambda x: rdp_bank.RDP_noisy_screen({'logp': params['logp'], 'logq': params['logq']}, x)
        self.propagate_updates(new_rdp, 'RDP')


class GaussianSVT_Mechanism(Mechanism):
    """
    The Gaussian-based SVT mechanism is described in
    https://papers.nips.cc/paper/2020/file/e9bf14a419d77534105016f5ec122d62-Paper.pdf

    The mechanism takes the parameter k and c. k is the maximum length before
    the algorithm stops. c is the cut-off parameter.  Setting rdp_c_1 = True implies that
    we use RDP-based Gaussian-SVT with c=1, else c>1.

    """
    def __init__(self,params,name='GaussianSVT', rdp_c_1=True):
        Mechanism.__init__(self)
        self.name=name
        if rdp_c_1 == True:
            self.name = name + 'c_1'
            valid_keys = ['sigma', 'k', 'margin', 'sigma_nu', 'Delta', 'sigma2', 'gamma']
            self.params = dict(filter(lambda tuple: tuple[0] in valid_keys, params.items()))
            new_rdp = lambda x: rdp_bank.RDP_gaussian_svt_c1(self.params, x)
        else:
            self.name = name + 'c>1'
            self.params = {'sigma':params['sigma'],'k':params['k'], 'c':params['c']}
            new_rdp = lambda x: rdp_bank.RDP_gaussian_svt_cgreater1(self.params, x)
        self.propagate_updates(new_rdp, 'RDP')

class LaplaceSVT_Mechanism(Mechanism):
    """
    Laplace SVT (c>=1) with a RDP description.
    b: the noise scale to perturb the threshold and the query.
    k: the maximum length before the algorithm steps. k could be infinite.

    We provide the RDP implementation and pure-DP implementation
    """
    def __init__(self,params,name='GaussianSVT'):
        Mechanism.__init__(self)
        self.name=name
        valid_keys = ['b', 'k', 'c']
        self.params = dict(filter(lambda tuple: tuple[0] in valid_keys, params.items()))
        new_rdp = lambda x: rdp_bank.RDP_svt_laplace(self.params, x)
        self.propagate_updates(new_rdp, 'RDP')


class StageWiseMechanism(Mechanism):
    """
    The StageWise generalized SVT is proposed by Zhu et.al., NeurIPS-20 (see Algorithm 3).
    This mechanism generalizes the Gaussian-based SVT.
    The mechanism takes two parameter c and k. c is the cut-off parameter (the number of tops in svt)
    and k denotes the  maximum limit for each chunk, e.g., the algorithm restarts whenever it encounters
     a top or reaches k limit.
    """
    def __init__(self, params=None,approxDP_off=False, name='StageWiseMechanism'):
        # the sigma parameter is the std of the noise divide by the l2 sensitivity
        Mechanism.__init__(self)

        self.name = name # When composing
        self.params = {'sigma': params['sigma'], 'k':params['k'], 'c':params['c']}
        self.delta0 = 0

        if not approxDP_off:  # Direct implementation of approxDP
            new_approxdp = lambda x: dp_bank.get_eps_gaussian_svt(params, x)
            self.propagate_updates(new_approxdp, 'approxDP_func')



# # Example 1: Short implementation of noisy gradient descent mechanism as a composition of GMs
# class NoisyGD_mech(GaussianMechanism):
#     def __init__(self,sigma_list,name='NoisyGD'):
#         GaussianMechanism.__init__(self, sigma=np.sqrt(np.sum(sigma_list)),name=name)
#         self.params = {'sigma_list':sigma_list}
#
# # The user could log sigma_list and then just declare a NoisyGD_mech object.
# mech = NoisyGD_mech(sigma_list)
# mech.get_approxDP(delta=1e-6)
#
#
# # Example 2: Implementing NoisySGD from basic building blocks
# subsample = Transformers.Subsample(prob=0.01)
# mech = Mechanisms.GaussianMechanism(sigma=5.0)
# # Create subsampled Gaussian mechanism
# SubsampledGaussian_mech = subsample(mech)
#
# # Now run this for 100 iterations
# compose = Transformers.Composition()
# NoisySGD_mech = compose(mechanism_list = [SubsampledGaussian_mech],coeffs_list=[100])
#
#
# # Example 3: You could also package this together by defining a NoisySGD mechanism
# class NoisySGD_mech(Mechanism):
#     def __init__(self,prob,sigma,niter,name='NoisySGD'):
#         Mechanism.__init__()
#         self.name=name
#         self.params={'prob':prob,'sigma':sigma,'niter':niter}
#
#         rdp = rdp_bank.subsampled_gaussian({'prob':params['prob'],'sigma':params['sigma']})
#         self.propagate_updates(rdp,type_of_update='RDP')
#
#
# # Example 4: Online decision. Hetereogenous sigma decided online
# # (maybe as a function of computed eps)
# # Alternatively if we want to do it via composition, so we can make online decision about
# # the sigma in the sigma_list
#
# delta = 1e-6
# online_sgd = Mechanisms.SubsampledGaussian_mech(prob=prob,sigma=sigma)
# compose = Transformers.Composition()
# for i in range(niter):
#     eps = online_ngd.get_approxDP(delta)
#     #determine the next prob, sigma
#     prob, sigma = func(eps)
#     mech_cur = Mechanisms.SubsampledGaussian_mech(prob=prob, sigma=sigma)
#     online_ngd = compose([online_ngd, mech_cur])
#
# # The above is quite general and can be viewed as a privacy accountant

class NoisySGD_Mechanism(Mechanism):
    def __init__(self, prob, sigma, niter, PoissonSampling=True, name='NoisySGD'):
        Mechanism.__init__(self)
        self.name = name
        self.params = {'prob': prob, 'sigma': sigma, 'niter': niter,
                       'PoissonSampling': PoissonSampling}

        # create such a mechanism as in previously
        subsample = transformer_zoo.AmplificationBySampling(PoissonSampling=PoissonSampling)
        # by default this is using poisson sampling

        mech = ExactGaussianMechanism(sigma=sigma)
        prob = prob
        # Create subsampled Gaussian mechanism
        SubsampledGaussian_mech = subsample(mech, prob, improved_bound_flag=True)
        # for Gaussian mechanism the improved bound always applies

        # Now run this for niter iterations
        compose = transformer_zoo.Composition()
        mech = compose([SubsampledGaussian_mech], [niter])

        # Now we get it and let's extract the RDP function and assign it to the current mech being constructed
        rdp_total = mech.RenyiDP
        self.propagate_updates(rdp_total, type_of_update='RDP')


class NoisyGD_Mechanism(GaussianMechanism):
    # With a predefined sequence of noise multipliers.
    def __init__(self,sigma_list, name='NoisyGD'):
        GaussianMechanism.__init__(self, sigma=np.sqrt(1/np.sum(1/sigma_list**2)), name=name)
        self.params = {'sigma_list': sigma_list}





