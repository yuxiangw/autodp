# This set of experiments compare subsampled gaussian of naive composition, CGF composition and asymp CGF composition

import math

from autodp.autodp import rdp_acct, rdp_bank, dp_acct, privacy_calibrator, utils

from scipy.optimize import minimize_scalar
import scipy.special
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.special
"""
This experiment compares the RDP \epsilon(alpha) for three subsample mechanisms, the range of 
alpha here is M. We generate alpha_list to compare it.
sigma, b, p : parameters for gaussian, laplace and random response
prob is the sample ratio, the default is 0.001
alpha_limit: computes the maximum available alpha for moment method in Abadi et. al.2016
approx_k_list: contains the parameter \tau(the approximate parameter in theorem 11 in ICML-19
acgfacct3:via tight upper/lower bound (Theorem 2/3)
general_acct: general upperbound, here we choose the approximate coeff for general upperbound to be 50

"""

epslist = []
m = 500  # for small \eps, m needs to be large for big \eps, m needs to be
delta = 1e-8
M = 100000 # for big \alpha
sigma = 5  # sigma for gaussian
b = 2  # gamma for laplace
p = 0.6 # parameter for random response
prob = 0.001  # sampling probability
approx_k_list = [2, 10, 50]
approx = []

# get the CGF functions
func_gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
func_laplace = lambda x: rdp_bank.RDP_laplace({'b': b}, x)
func_randresp = lambda x: rdp_bank.RDP_randresponse({'p': p}, x)

func_asymp = lambda x: rdp_bank.pRDP_asymp_subsampled_gaussian({'sigma': sigma, 'prob': prob}, x)

# The best possible data dependent bound when prob * n  = 100
func_asymp_best = lambda x: rdp_bank.pRDP_asymp_subsampled_gaussian_best_case(
    {'sigma': sigma, 'prob': prob, 'n': 100 / prob}, x)

funcs = {'laplace': func_laplace,'gaussian': func_gaussian, 'randresp': func_randresp}

# A placeholder to save results for plotting purposes
results_for_figures = {}



def get_moment(alpha, sigma, gamma):
    t = 3
    part_1 = gamma ** 2 * alpha / ((1 - gamma) * sigma ** 2)
    part_2 = (2 * gamma) ** t / ((1 - gamma) ** t * sigma ** (2 * t))
    part_3 = (2 * gamma) ** t * (t - 1) / (2 * (1 - gamma) ** (t - 1) * sigma ** t)
    part_4 = (2 * prob) ** t * np.exp((t ** 2 - t) / (2 * sigma ** 2)) * (sigma ** t * 2 + t ** t) / (
                2 * (1 - gamma) ** (t - 1) * sigma ** (2 * t))
    bound = part_1 + alpha * (alpha - 2) / 6 * (part_2 + part_3 + part_4)
    part_sum = 4 * alpha * (alpha - 2) * gamma ** 3 / (3 * sigma ** 3)

    return bound



dense = 1.07
alpha_list = [int(dense ** i + 1) for i in range(int(math.floor(math.log(M, dense))) + 1)]
alpha_list = np.unique(alpha_list)


for name, func in funcs.items():
    figure_2 = []
    cgf_poisson = []
    # Declare the analytical CGF accountant
    acgfacct = rdp_acct.anaRDPacct(m=m, m_max = 1000,m_lin_max=M)

    # Declare another analytical CGF accountant for calculating the lower bound
    acgfacct3 = rdp_acct.anaRDPacct(m=m,m_max = 1000,m_lin_max=M)
    # general_acct tracks the general upperbound, we set approx=True for approximate methods
    general_acct =rdp_acct.anaRDPacct(m=m,m_max= 1000,m_lin_max=M, approx=True)

    def cgf(x):
        return (x - 1) * func(x)


    moment = []  # only for gaussian

    if name == 'gaussian':
        moment = [get_moment(mm, sigma, prob) for mm in alpha_list]



    acgfacct.compose_subsampled_mechanism(func, prob)
    acgfacct3.compose_poisson_subsampled_mechanisms(func, prob)
    figure_2.append(acgfacct3)
    for kk in approx_k_list:

        fast_k_approx = rdp_acct.anaRDPacct(m=kk, m_max=kk, m_lin_max=2,approx=True)
        fast_k_approx.compose_poisson_subsampled_mechanisms(func, prob)
        figure_2.append(fast_k_approx)

    general_acct.compose_subsampled_mechanism(func,prob)
    results_for_figures[name] = {
    "fast_k": [x.get_rdp([i for i in alpha_list]) for x in figure_2],
    "general": [general_acct.get_rdp([i for i in alpha_list]), acgfacct.get_rdp([i for i in alpha_list]),
                acgfacct3.get_rdp([i for i in alpha_list])]
    }
    print('tight={} fast={}'.format(results_for_figures[name]['fast_k'][0][:20],results_for_figures[name]['fast_k'][1][:20]))
    font = {'family': 'times',
            'weight': 'bold',
            'size': 13}

    matplotlib.rc('font', **font)


    plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.loglog(alpha_list, acgfacct3.get_rdp([i for i in alpha_list]), '-k',
               linewidth=1)
    plt.loglog(alpha_list, figure_2[1].get_rdp([i for i in alpha_list]), '-r', linewidth=1)
    plt.loglog(alpha_list,figure_2[2].get_rdp([i for i in alpha_list]) , '--b', linewidth=2)
    plt.loglog(alpha_list, figure_2[3].get_rdp([i for i in alpha_list]), '--g', linewidth=2)  # 'subsampled poisson lower bound',

    plt.legend(
        [r'Tight upper/lower bound (Theorem 2/3)', r'Fast_approximation_$ \tau =2$', r'Fast_approximation $\tau =10$',
         r'Fast_approximation $\tau =50$'], loc='lower right')
    if name == 'laplace':
        plt.title(r'Subsampled Laplace with $b=2, \gamma = 0.001$')
    elif name == 'randresp':
        plt.title(r'Subsampled random response with $p=0.6, \gamma = 0.001$ ')
    elif name == 'gaussian':
        plt.title(r'Subsampled gaussian with $\sigma=5, \gamma = 0.001$')
    plt.xlabel(r'RDP order $\alpha$')
    plt.ylabel(r'RDP $\epsilon(\alpha)$')
    plt.grid(True)
    plt.savefig(name + "_high" + "_approx.pdf", bbox_inches='tight')

    plt.show()

    plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

    alpha_limit = int(sigma ** 2 * np.log(1 / (prob * sigma)))
    alpha_list = np.array(alpha_list)
    max_alpha = np.where(alpha_list > alpha_limit)
    plt.loglog(alpha_list, general_acct.get_rdp([i for i in alpha_list]), '--r', linewidth=2)

    plt.loglog(alpha_list, acgfacct.get_rdp([i for i in alpha_list]), '-b',
               linewidth=2)
    plt.loglog(alpha_list, acgfacct3.get_rdp([i for i in alpha_list]), '--k',
               linewidth=2)
    if name == 'gaussian':
        plt.plot(alpha_list[0:max_alpha[0][0]], moment[0:max_alpha[0][0]], '-g', linewidth=2)
    plt.legend(['General upper bound (Theorem 1)', 'Sample w/o Replacement [Theorem 27, Wang et al., 2019]',
                'Tight upper/lower bound (Theorem 2/3)', 'Bound for Gaussian [Lemma 3, Abadi et. al., 2016]'],
               loc='lower right')
    if name == 'laplace':
        plt.title(r'Subsampled Laplace with $b=2, \gamma = 0.001$ ')
    elif name == 'randresp':
        plt.title(r'Subsampled random response with $p=0.6, \gamma = 0.001$')
    elif name == 'gaussian':
        plt.title(r'Subsampled gaussian with $\sigma=5, \gamma = 0.001 $ ')
    plt.xlabel(r'RDP order $\alpha$')
    plt.ylabel(r'RDP $\epsilon(\alpha)$')
    plt.grid(True)
    plt.savefig(name + "_high_general.pdf", bbox_inches='tight')

    plt.show()

import pickle

file_Name = "high_privacy.results"
# open the file for writing
fileObject = open(file_Name, 'wb')

# this writes the object a to the
# file named 'testfile'
pickle.dump(results_for_figures, fileObject)

# here we close the fileObject
fileObject.close()
