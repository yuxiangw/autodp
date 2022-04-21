from autodp.mechanism_zoo import LaplaceMechanism, RandresponseMechanism
from autodp.transformer_zoo import Composition, ComposeAFA, ComposeGaussian, AmplificationBySampling
import matplotlib.pyplot as plt
from autodp.cdf_bank import cdf_approx_fft
import numpy as np
from autodp.phi_bank import phi_laplace, phi_gaussian, phi_rr_p

eps = 0.01
delta = 1e-6
laplace_phi = LaplaceMechanism( b=1./eps, phi_off=False)

laplace_rdp = LaplaceMechanism(b=1./eps)
rr_phi = RandresponseMechanism(p = 1.0/(np.exp(eps)+1), phi_off=False)
compose_afa = ComposeAFA()
compose_rdp = Composition()
eps_rdp = []
eps_phi = []
eps_rr = []

def test_comp():
    klist = [int(10*i) for i in range(2,10)]
    print(klist)
    for k in klist:
        print('k', k)
        cur_composed_phi = compose_afa([laplace_phi], [k])
        cur_composed_rdp = compose_rdp([laplace_rdp], [k])
        cur_composed_rr = compose_afa([rr_phi], [k])
        eps_rdp.append(cur_composed_rdp.approxDP(delta))
        print('eps_rdp', eps_rdp)
        eps_phi.append(cur_composed_phi.approxDP(delta))
        print('eps_quad', eps_phi)
        eps_rr.append(cur_composed_rr.approxDP(delta))
        #eps_phi.append(cur_composed_phi.approx_delta(delta))

    print('eps_rdp', eps_rdp)
    print('eps_phi', eps_phi)

    plt.figure(figsize = (6,6))

    plt.plot(klist, eps_rdp , 'm', linewidth=2)
    plt.plot(klist, eps_phi, 'D--', color = 'pink', linewidth=2)
    plt.plot(klist, eps_rr, 'k.-.',  linewidth=2)
    plt.yscale('log')
    plt.legend(
        [r'Laplace-RDP','Laplace-AFA', 'RR-AFA'], loc='best', fontsize=18)
    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'Number of Compositions $k$', fontsize=18)
    plt.ylabel(r'$\epsilon$', fontsize=18)
    plt.title(r'Laplace mechanism eps is'+str(eps))
    plt.show()

test_comp()

def plot_cdf_lap():
    """
    the x axis is the range of log(p/q) (log_e), also known as epsilon.
    delta = 1-(cdf_p(log_p/q) + x*cdf_q(-log_p/q))
    """
    # eps denotes log_p/q
    eps = [i for i in range(1, 100)]
    #params = {'b': 10.}
    params = {'sigma':10.}
    log_phi = lambda l: phi_gaussian(params, l)
    #log_phi = lambda l: phi_laplace(params, l)
    from autodp.cdf_bank import cdf_approx
    cdf_p = [cdf_approx(log_phi, log_p, tbd=None) for log_p in eps]

    print('cdf_p', cdf_p)

    plt.figure(figsize = (6,6))

    plt.plot(eps, cdf_p, 'D--', color = 'pink', linewidth=2)

    plt.yscale('log')
    plt.legend(
        [r'cdf'], loc='best', fontsize=18)
    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'range of log(p/q)', fontsize=18)
    plt.ylabel(r'CDF', fontsize=18)
    plt.title(r'the CDF of Laplace mechanism, $b=0.5$')
    plt.show()

#plot_cdf_lap()
def test_lap():
    # print the distribution of phi(t), x-axis is t, y axis is phi-t.
    x = [1.5**i for i in range(-100, 100)]
    eps = 5.0
    print('x', x)
    params = {'b':1./eps}
    params = {'sigma':1.0}
    p = 1.0/(np.exp(eps)+1)
    #log_phi = lambda l: phi_laplace(params, l)
    #log_phi = lambda l: phi_gaussian(params, l)
    log_phi =  lambda x: phi_rr_p({'p': p, 'q':1-p}, x)
    #cdf_infty =  cdf_approx_fft(log_phi, 2000000)
    #cdf = cdf_approx_fft(log_phi, eps)
    #print('cdf_infty', cdf_infty, 'cdf current', cdf)
    phi_list = [log_phi(t)  for t in x]
    #phi_list_a = np.array([log_phi(t)+np.log(1.j)-1.j*t*eps - np.log(t) for t in x])
    #phi_list_b = np.array([log_phi(-t) + np.log(1.j) +1.j * t * eps - np.log(-t) for t in x])
    #phi_list = phi_list_a + phi_list_b
    #phi_list_2 = [log_phi_p(t)+np.log(1.j)-1.j*t*eps*10 - np.log(t) for t in x]
    #diff = [i-j for (i,j) in zip(phi_list, phi_list_2)]
    #print('phi_list', diff)
    print('diff', phi_list)

    plt.figure(figsize=(6, 6))


    plt.plot(x, phi_list, 'm', linewidth=2)


    #plt.yscale('log')
    plt.xscale('log')

    plt.legend(
        [r'log of phi', 'log q'], loc='best', fontsize=18)
    plt.grid(True)
    plt.xscale('log')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'Range of  $t$', fontsize=18)
    plt.ylabel(r'$\Phi(t)$', fontsize=18)
    plt.title(r'Laplace mechanism, $b=0.5$')
    plt.show()
test_lap()