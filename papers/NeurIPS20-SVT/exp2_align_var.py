import numpy as np
import math
import scipy
from autodp import rdp_bank, dp_bank, fdp_bank, utils
from autodp.mechanism_zoo import LaplaceMechanism, LaplaceSVT_Mechanism,StageWiseMechanism
from autodp.transformer_zoo import Composition
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace
from scipy.special import comb
#import matplotlib.font_manager as fm
from autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism,SubsampleGaussianMechanism, GaussianMechanism, ComposedGaussianMechanism,GaussianSVT_Mechanism, NoisyScreenMechanism
from autodp.transformer_zoo import Composition, AmplificationBySampling


"""
This experiment corresponding to exp 2 in NeurIPS-20 (Figure 2 (a))
We evaluate SVT variants with the same variance of noise by comparing the composed privacy loss for ﬁnishing a ﬁxed length sequence of queries.

rho is from Lap(lambda) -> eps_rho = 1/lambda
nu is from Lap(2lambda) -> eps_nu = 1/lambda
eps = (c+1)/lambda, lambda = (c+1)/eps

To align variance between Gaussian-bassed and Laplace-based approaches, we set sigma_1 = sqrt(2) * lambda_rho
"""

delta = 1e-6
lambda_rho = 120
lambda_nu = 240
# Under this setting, sigma_1 / L2 sensitivity = sigma_2 /2 * L2 sensitivity
sigma_1 = lambda_rho*np.sqrt(2)
sigma_2 = 2*sigma_1
eps_1 = 1.0 / lambda_rho
n = 100000 #the length of the fixed query
margin = 1000



def exp_2a():

    eps_a = []   #standard SVT
    eps_e = []   #Laplace-SVT (via RDP)
    eps_g = []   #Gaussian SVT c>1
    eps_g_c = [] #c=1 for Gaussian SVT
    eps_i = []
    eps_kov = [] #generalized SVT
    eps_noisy = []
    k_list =  [int(1.4**i) for i in range(int(math.floor(math.log(n,1.4)))+1)]
    print(len(k_list))

    query = np.zeros(n)
    rho = np.random.normal(scale=sigma_1)
    lap_rho = np.random.laplace(loc=0.0, scale=lambda_rho)

    """
    compute eps for noisy screening
    p = Prob[ nu > Margin]
    q = Prob[ nu - 1 > margin]
    
    count_gau counts #tops in Gaussian-SVT
    count_lap counts #tops in Laplace-SVT
    """

    count_gau = 0
    count_lap = 0

    # the following is for data-dependent screening in CVPR-20
    p = scipy.stats.norm.logsf(margin, scale=sigma_2)
    q = scipy.stats.norm.logsf(margin + 1, scale=sigma_2)
    params = {}
    params['logp'] = p
    params['logq'] = q
    per_screen_mech = NoisyScreenMechanism(params, name='NoisyScreen')
    per_gaussian_mech = ExactGaussianMechanism(sigma_2,name='GM1')
    index = []
    compose = Composition()
    for idx, qu in enumerate(query):
        nu = np.random.normal(scale=sigma_2)
        lap_nu = np.random.laplace(loc=0.0, scale=lambda_nu)
        if nu >= rho + margin:
            count_gau += 1
        if lap_nu >= lap_rho + margin:
            count_lap += 1
        count_gau = max(count_gau, 1)
        count_lap = max(count_lap, 1)

        if idx in k_list:
            index.append(idx)
            print('number of queries passing threshold', count_gau)

            #eps_a records the standard SVT
            eps_a.append(eps_1 * count_lap + eps_1)
            # compose data-dependent screening
            screen_mech = compose([per_screen_mech], [idx])
            gaussian_mech = compose([per_gaussian_mech], [idx])

            # standard SVT with RDP calculation
            param_lap_svt = {}
            param_lap_svt['b'] = lambda_rho
            param_lap_svt['k'] = idx
            param_lap_svt['c'] = count_lap
            lapsvtrdp_mech = LaplaceSVT_Mechanism(param_lap_svt)
            eps_e.append(lapsvtrdp_mech.get_approxDP(delta))

            # stage-wise generalized SVT, k is the maximum length of each chunk
            k = int(idx / np.sqrt(count_gau))
            # assume sensitivity is 1, sigma_nu denotes noise add to query / 2
            generalized_mech = StageWiseMechanism({'sigma':sigma_1,'k':k, 'c':count_gau, 'delta':delta})
            eps_kov.append(generalized_mech.get_approxDP(delta))

            # Gaussian-SVT c>1 with RDP, k is the total length before algorithm stops
            gaussianSVT_c = GaussianSVT_Mechanism({'sigma':sigma_1,'sigma_nu':sigma_1,'k':idx, 'c':count_gau}, rdp_c_1=False)
            eps_g.append(gaussianSVT_c.get_approxDP(delta))

            #Gaussian-SVT with c=1, we use average_k as the approximate maximum length of each chunk, margin is used in Proposition 10
            average_k = int(idx / max(count_gau, 1))
            params_SVT = {}
            params_SVT['k'] = average_k
            params_SVT['sigma'] = sigma_1
            params_SVT['sigma_nu'] = sigma_1
            params_SVT['margin'] = margin
            per_gaussianSVT_mech = GaussianSVT_Mechanism(params_SVT)
            gaussianSVT_mech = compose([per_gaussianSVT_mech],[max(count_gau, 1)])
            eps_g_c.append(gaussianSVT_mech.get_approxDP(delta))

            eps_i.append(gaussian_mech.get_approxDP(delta))  # Gaussian Mechanism
            eps_noisy.append(screen_mech.get_approxDP(delta))

    import matplotlib
    import matplotlib.pyplot as plt

    font = {'family': 'times',
            'weight': 'bold',
            'size': 18}

    #props = fm.FontProperties(family='Gill Sans', fname='/Library/Fonts/GillSans.ttc')
    f, ax = plt.subplots()
    plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.loglog(index, eps_a, '-r', linewidth=2)
    plt.loglog(index, eps_e, '--g^', linewidth=2)
    plt.loglog(index, eps_g, '-c^', linewidth=2)
    plt.loglog(index, eps_g_c, '-bs', linewidth=2)
    plt.loglog(index, eps_i, '--k', linewidth=2)
    plt.loglog(index, eps_noisy, color='brown', linewidth=2)
    plt.loglog(index, eps_kov, color='hotpink', linewidth=2)
    plt.legend(
        ['Laplace-SVT (Pure-DP from Lyu et al., 2017)', 'Laplace-SVT (via RDP)', 'Gaussian-SVT c>1 (RDP by Theorem 11)',
         'Gaussian-SVT c=1 (RDP by Theorem 8)', 'Gaussian Mechanism', 'Noisy Screening (data-dependent RDP)',
         'Stage-wise generalized SVT'], loc='best', fontsize=17)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'Iterations', fontsize=20)
    plt.ylabel(r'$\epsilon$', fontsize=20)
    #ax.set_title('Title', fontproperties=props)
    plt.show()
    #plt.savefig('exp2a.pdf', bbox_inches='tight')
exp_2a()
