# This set of experiments compare our bounds in moments accuntant, corresponding to figure 4 in ICML-19
# We compare our methods with naive composition, CGF composition

import math
import time
from autodp import rdp_acct, rdp_bank, dp_acct,privacy_calibrator,utils
import numpy as np
from scipy.optimize import minimize_scalar
import pickle
import matplotlib
import matplotlib.pyplot as plt
from  scipy.special import binom as bin
epslist = []

"""
k is the number of iteration
sigma, b, p : parameters for gaussian, laplace and random response
prob is the sample ratio
alpha_limit computes the maximum available alpha for moment method in Abadi et. al.2016
We declare 6 moment account here
acgfacct: Sample w/o Replacement [WBK’18]
acgfacct3:via tight upper/lower bound (Theorem 2/3)
acgfacct5: general upperbound
eps_seq_simple: naive composition
eps_seq_naive: strong composition [Kairouz et al.KOV15]
moment_cache: moment method in Abadi et. al.2016
We compare their privacy loss after k's iteration

"""

m=100 # for small \eps, m needs to be large for big \eps, m needs to be
delta = 1e-8
cgfacct = rdp_acct.anaRDPacct(m)
k= 60000
sigma = 5
b = 2
p = 0.6
prob=0.001 # sampling probability
alpha_limit = int(sigma ** 2 * np.log(1 / (prob * sigma))) # maximum alpha for moment method



def naive_eps_simple(func, prob, k, delta): # naive simple composition
    # input x is log delta, it needs to be negative
    tmp_acct = rdp_acct.anaRDPacct()
    tmp_acct.compose_mechanism(func)
    eps = tmp_acct.get_eps(delta/k/prob)
    eps1, delta1 = rdp_acct.subsample_epsdelta(eps, delta/k/prob, prob)

    return eps1*k

def naive_eps(x, func, prob, k, delta): # naive strong composition
    # input x is log delta, it needs to be negative
    #t1 = time.time()
    tmp_acct = rdp_acct.anaRDPacct()
    tmp_acct.compose_mechanism(func)
    eps = tmp_acct.get_eps(np.exp(x))
    eps1, delta1 = rdp_acct.subsample_epsdelta(eps, np.exp(x), prob)
    eps_1 = k*eps1
    deltatilde = 1 - np.exp(np.log(1-delta) - k*np.log(1-delta1))
    eps_2 = k*eps1**2+ eps1 *(2*k*np.log(np.exp(1) +(k*eps1**2)**0.5/deltatilde))**0.5
    eps_3 = k*eps1**2 + eps1*(2*k*np.log(1 / deltatilde))**0.5
    eps_all = np.min([eps_1,eps_2,eps_3])

    if eps_all < 0:  # it will be -1
        return np.inf
    else:
        return eps_all

#const = 1000000
const =10
def smallest_eps(func, prob, k, delta):
    fun = lambda x: naive_eps(x, func, prob, k, delta)

    results = minimize_scalar(fun, method='Bounded',
                              bounds=[np.log(delta / k / prob / const), np.log(delta / k / prob * 2)],
                              options={'disp': False})
    return results.fun, results.x



def get_moment(alpha, sigma, gamma):

    t = 3
    part_1 = gamma**2*alpha/((1-gamma)*sigma**2)
    part_2 = (2*gamma)**t/((1-gamma)**t*sigma**(2*t))
    part_3 = (2*gamma)**t*(t-1)/(2*(1-gamma)**(t-1)*sigma**t)
    part_4 = (2*prob)**t*np.exp((t**2-t)/(2*sigma**2))*(sigma**t*2+t**t)/(2*(1-gamma)**(t-1)*sigma**(2*t))
    bound = part_1 + alpha*(alpha-2)/6*(part_2+part_3+part_4)
    part_sum = 4 * alpha * (alpha - 2) * gamma ** 3 / (3 * sigma ** 3)
    return bound
# get the CGF functions
func_gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
func_laplace = lambda x: rdp_bank.RDP_laplace({'b': b}, x)
func_randresp = lambda x: rdp_bank.RDP_randresponse({'p':p}, x)
func_moment = lambda x: get_moment({'gamma':prob,'sigma':sigma},x)
funcs = { 'gaussian':func_gaussian,'laplace': func_laplace, 'randresp': func_randresp}

# A placeholder to save results for plotting purposes
results_for_figures = {}

# compute feasible alpha list for moment account in Abadi et. al.2016
moment =[]
moment_alpha =np.linspace(1, alpha_limit, alpha_limit).astype(int)
for mm in moment_alpha:
    moment.append(get_moment(mm, sigma, prob))
moment_cache = np.zeros(alpha_limit)
moment = np.array(moment)
def get_eps_moment(alpha,cache):

    return np.min(np.log(1 / delta) / (alpha[1:] - 1) + cache[1:])

for name, func in funcs.items():


    #Declare the analytical moment accountant
    acgfacct = rdp_acct.anaRDPacct(m=m)

    # Declare another analytical CGF accountant for calculating the lower bound
    acgfacct3 = rdp_acct.anaRDPacct(m=m)
    acgfacct5 = rdp_acct.anaRDPacct(m=m, approx=True)

    eps_seq = []
    eps_seq_simple = []
    eps_seq_naive = []
    eps_seq_lowerbound = []
    eps_seq_kapprox = []
    eps_seq_general = []
    eps_moment = []



    seq_opt_x = []
    klist = [2**i for i in range(int(math.floor(math.log(k,2)))+1)]


    for i in range(k):
        acgfacct.compose_subsampled_mechanism(func,prob)
        acgfacct3.compose_poisson_subsampled_mechanisms(func,prob)
        acgfacct5.compose_subsampled_mechanism(func, prob)
        moment_cache = moment_cache+ moment
        if i+1 in klist:
            eps_seq.append(acgfacct.get_eps(delta))
            t1 = time.time()
            eps1, x = smallest_eps(func, prob, i + 1, delta)
            eps_seq_naive.append(eps1)
            eps_seq_simple.append(naive_eps_simple(func, prob, i + 1, delta))
            eps_seq_lowerbound.append(acgfacct3.get_eps(delta))
            eps_seq_general.append(acgfacct5.get_eps(delta))
            eps_moment.append(get_eps_moment(moment_alpha,moment_cache))

    lamblist = [2**i+1 for i in range(int(math.floor(math.log(263000,2)))+1)]
    lamblist1 = np.array(lamblist)





    import matplotlib
    import matplotlib.pyplot as plt

    font = {'family': 'times',
            'weight': 'bold',
            'size': 14}

    matplotlib.rc('font', **font)



    plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.loglog(klist, eps_seq_simple, '-c^',linewidth=2)
    plt.loglog(klist, eps_seq_naive, '-r^', linewidth=2)
    plt.loglog(klist, eps_seq_general,'-g^',linewidth=2)
    plt.loglog(klist, eps_seq,'-bs',linewidth=2)
    plt.loglog(klist, eps_seq_lowerbound,'--k',linewidth=2)
    if name == 'gaussian':
        plt.loglog(klist, eps_moment,'--r',linewidth=2)
    plt.grid(True)

    plt.xlabel(r'iteration $k$')
    plt.ylabel(r'$\epsilon$')

    if func is func_gaussian:
        plt.legend([r' naive composition','strong composition [Kairouz et al.KOV15]','via general upper bound (Theorem 1)','via Sample w/o Replacement [WBK’18]','via tight upper/lower bound (Theorem 2/3)','via bound for Gaussian [Lemma2, Abadi et. al.,2016]'], loc='best')
    else:
        plt.legend([r'naive composition', 'strong composition [Kairouz et al.KOV15]','via general upper bound (Theorem 1)',r'via Sample w/o replacement [WBK’18]',  'via tight upper/lower bound (Theorem 2/3)'],loc='best')

    if name == 'laplace':
        plt.title(r'Overall (eps,delta)-DP over composition for Subsampled Laplace with $b=0.5, \gamma = 0.001$')
    elif name =='randresp':
        plt.title(r'Overall (eps,delta)-DP over composition for Subsampled random response with $p=0.9, \gamma = 0.001$')
    elif name == 'gaussian':
        plt.title(r'Overall (eps,delta)-DP over composition for Subsampled gaussian with $\sigma=1, \gamma = 0.001$')
    plt.show()
    plt.savefig(name+"high_eps.pdf",bbox_inches='tight')

    plt.close('all')




import pickle

file_Name = "normal_anaCGFacct.results"
# open the file for writing
fileObject = open(file_Name, 'wb')

# this writes the object a to the
# file named 'testfile'
pickle.dump(results_for_figures, fileObject)

# here we close the fileObject
fileObject.close()

