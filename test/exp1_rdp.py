import numpy as np
from autodp.mechanism_zoo import GaussianMechanism, ExactGaussianMechanism, PureDP_Mechanism, LaplaceMechanism,SubSampleGaussian
import math
import matplotlib.font_manager as fm
from autodp.transformer_zoo import Composition, AmplificationBySampling
import os
import pickle

def exp_1_fdp():
    # Example 1: Gaussian mechanism
    sigma = 2.0


    gm0 = GaussianMechanism(sigma,name='GM0',approxDP_off=True, use_basic_RDP_to_approxDP_conversion=True)
    gm1 = GaussianMechanism(sigma,name='GM1',approxDP_off=True)
    gm1b = GaussianMechanism(sigma,name='GM1b',approxDP_off=True, use_fDP_based_RDP_to_approxDP_conversion=True)
    gm2 = GaussianMechanism(sigma,name='GM2',RDP_off=True)
    gm3 = GaussianMechanism(sigma,name='GM3',RDP_off=True, approxDP_off=True, fdp_off=False)



    eps = np.sqrt(2)/sigma # Aligning the variance of the laplace mech and gaussian mech
    laplace = PureDP_Mechanism(eps,name='Laplace')

    label_list = ['naive_RDP_conversion','BBGHS_RDP_conversion','Our new method',
                  'exact_eps_delta_DP','exact_fdp',r'laplace mech ($b = \sqrt{2}/\sigma$)']


    import matplotlib.pyplot as plt



    fpr_list, fnr_list = gm0.plot_fDP()
    fpr_list1, fnr_list1 = gm1.plot_fDP()
    fpr_list1b, fnr_list1b = gm1b.plot_fDP()
    fpr_list2, fnr_list2 = gm2.plot_fDP()
    fpr_list3, fnr_list3 = gm3.plot_fDP()
    fpr_list4, fnr_list4 = laplace.plot_fDP()

    plt.figure(figsize=(4,4))
    plt.plot(fpr_list,fnr_list)
    plt.plot(fpr_list1,fnr_list1)
    plt.plot(fpr_list1b,fnr_list1b)
    plt.plot(fpr_list2, fnr_list2)
    plt.plot(fpr_list3, fnr_list3,':')
    plt.plot(fpr_list4, fnr_list4,'-.')
    plt.legend(label_list)
    plt.xlabel('Type I error')
    plt.ylabel('Type II error')
    plt.savefig('rdp2fdp.pdf')
    plt.show()

def exp2_gaussian():
    """
    Compare privacy cost over composition, y-axis is delta, epsilon is fixed.
    """


    #the number of compositions
    klist = [80* i for i in range(2,15)][3:]

    print('coeff list', klist)
    doc = {}
    eps = 2.0
    for sigma in [30, 60]:
        # RDP
        eps_rdp = []
        # Analytical Gaussian
        eps_exact = []
        # phi function
        eps_phi = []
        for coeff in klist:
            rdp_gaussian = GaussianMechanism(sigma, name='Laplace')
            exact_gaussian= GaussianMechanism(sigma,coeff= coeff, CDF_off=False)
            phi_gaussian = GaussianMechanism(sigma,coeff= coeff, CDF_off=False, cdf_approx=True)
            compose = Composition()
            composed_rdp_gaussian = compose([rdp_gaussian], [coeff])
            eps_rdp.append(composed_rdp_gaussian.approx_delta(eps))
            eps_exact.append(exact_gaussian.approx_delta(eps))
            eps_phi.append(phi_gaussian.approx_delta(eps))

        print('delta with analytical Gaussian mechanism', exact_gaussian)
        print('delta with RDP analysis', eps_rdp)
        print('delta with phi function method', eps_phi)
        cur_result = {}
        cur_result['rdp'] = eps_rdp
        cur_result['gt'] = eps_exact
        cur_result['phi'] = eps_phi
        doc[str(sigma)] = cur_result
    #with open(path, 'wb') as f:
    #    pickle.dump(doc, f)


    import matplotlib.pyplot as plt

    props = fm.FontProperties(family='Gill Sans', fname='/Library/Fonts/GillSans.ttc')
    f, ax = plt.subplots()
    plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(klist, doc['30']['rdp'], 'm', linewidth=2)
    plt.plot(klist, doc['30']['gt'], 'D--', color = 'pink', linewidth=2)
    plt.plot(klist, doc['30']['phi'], 'x-',  color ='darkred',linewidth=2)
    plt.plot(klist, doc['60']['rdp'],  color='darkorange', linewidth=2)
    plt.plot(klist, doc['60']['gt'], 'D--', color='pink', linewidth=2)
    plt.plot(klist, doc['60']['phi'], 'x-', color='darkblue',linewidth=2)
    plt.yscale('log')
    plt.legend(
        [r'RDP $\sigma=30$','Exact Accountant $\sigma=30$' ,'Our AFA $\sigma=30$','RDP $\sigma=60$','Exact Accountant $\sigma=60$','Our AFA with $\sigma=60$'], loc='best', fontsize=18)
    plt.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel(r'Number of Compositions $k$', fontsize=22)
    plt.ylabel(r'$\delta$', fontsize=22)
    ax.set_title('Title', fontproperties=props)
    #plt.show()
    plt.savefig('exp1_delta.pdf', bbox_inches='tight')


#exp2_gaussian()


def exp4_subsample_fixed_eps():
    """
    Evaluate poisson subsample Gaussian mechanism
    sample probability = 0.02, fix epsilon to 1.0 and compare delta over composition
    x axis is # composition
    y axis is delta(epsilon)

    Evaluate four methods
    BBGHS_RDP :eps_rdp
    Our phi-function lower bound: eps_phi_left
    Our phi-function upper bound:  eps_phi_right
    Double quadrature:eps_quadrature
    """

    eps = 1.0

    import pickle
    prob = 0.02
    klist = [100* i for i in range(2,16)]
    exp4_path = 'gamma_0.02_sigma_12.pkl'
    if os.path.exists(exp4_path):
        with open(exp4_path, 'rb') as f:
            doc = pickle.load(f)
            klist = klist[:]
            eps_phi_left = doc['phi_left']
            eps_phi_right = doc['phi_right']
            eps_rdp = doc['rdp']
            eps_quarture = doc['quarture']
    else:

        for sigma in [2]:
            eps_rdp = []
            eps_phi_left = []
            eps_phi_right = []
            eps_quarture = []
            for coeff in klist:
                gm1 = ExactGaussianMechanism(sigma, name='GM1')
                compose = Composition()
                poisson_sample = AmplificationBySampling(PoissonSampling=True)
                composed_mech = compose([poisson_sample(gm1, prob, improved_bound_flag=True)], [coeff])
                #uncomment the code below to run upper and lower bounds, which is much slower
                #phi_subsample_left = SubSampleGaussian(sigma, prob,coeff, CDF_off=False, lower_bound = True)
                #phi_subsample_right = SubSampleGaussian(sigma, prob,coeff, CDF_off=False, upper_bound = True)
                phi_quarture =  SubSampleGaussian(sigma, prob,coeff, CDF_off=False)
                eps_rdp.append(composed_mech.approx_delta(eps))
                eps_quarture.append(phi_quarture.get_approx_delta(eps))
                #eps_phi_left.append(phi_subsample_left.approx_delta(eps))
                #eps_phi_right.append(phi_subsample_right.approx_delta(eps))
                print('eps using double quarture', eps_quarture)
                print('eps using phi lower bound', eps_phi_left)
                print('eps using phi right bound', eps_phi_right)
                print('eps using rdp', eps_rdp)
            cur_result = {}
            cur_result['rdp'] = eps_rdp

            #eps_phi = [8.126706321817492e-11, 1.680813696457811e-08, 3.7171456678093376e-07, 2.8469414374309687e-06, 1.2134517188197369e-05, 3.612046366420398e-05, 8.487142126792672e-05, 0.00016916885963894502]
            # previous experimental records for the sigma = 2.0, eps =1.0
            cur_result['phi_left']=  [1.9705163375286992e-11, 6.54413434131078e-09, 1.8172415920878554e-07, 1.5912195465344071e-06, 7.408024600416465e-06, 2.3476597821679127e-05, 5.780615396550352e-05, 0.0001194871003954497,0.00021761168363315566, 0.00036048564689211086, 0.0005551755399635068, 0.0008073269983187659, 0.0011211655514957834, 0.0014996039591825359]
            cur_result['phi_right'] = [1.4208450705963226e-10, 2.7065873453364222e-08, 5.615768975610223e-07, 4.101636149422114e-06, 1.6856443667586166e-05, 4.875023579764651e-05, 0.00011190249522187185, 0.00021878046437442066,0.00038078926980138466, 0.0006074760381444841, 0.0009062357815770332, 0.0012823422696145335, 0.0017391548126395213, 0.0022783994671059515]
            #cur_result['phi_left'] = eps_phi_left
            #cur_result['phi_right'] = eps_phi_right
            cur_result['quarture'] = eps_quarture

            with open(exp4_path, 'wb') as f:
                pickle.dump(cur_result, f)
    # copy the results from Fourier accountant paper
    fft_lower =[7.887405682751439e-11, 1.621056159906838e-08, 3.547744440606434e-07, 2.6891898301208684e-06, 1.1344617219024322e-05, 3.342412117224878e-05, 7.773514229351955e-05, 0.00015336724020791973, 0.00026855202090630444, 0.00042999375546865035, 0.0006426099200177218, 0.0009095489321732455, 0.00123236372576394, 0.0016112544734264512]
    fft_higher =[8.195861133559092e-11, 1.7175792871794577e-08, 3.8343699421628997e-07, 2.9650469214763064e-06, 1.2761325748530785e-05, 3.836009767606709e-05, 9.10255425945141e-05, 0.0001832372011979862, 0.00032737944462639086, 0.0005348503868262945, 0.0008155844687884689, 0.0011778834943758018, 0.001628443617837156, 0.002172490405255]
    for sigma in [2]:

        eps_optimal_rdp = []

        for coeff in klist:
            gm1 = ExactGaussianMechanism(sigma, name='GM1')
            compose = Composition()
            poisson_sample = AmplificationBySampling(PoissonSampling=True)
            composed_mech = compose([poisson_sample(gm1, prob, improved_bound_flag=True)], [coeff])
            eps_optimal_rdp.append(composed_mech.approx_delta(eps))

    print('optimal conversion', eps_optimal_rdp)
    print('quarture result', eps_quarture)
    print('RDP result', eps_rdp)
    import matplotlib.pyplot as plt

    props = fm.FontProperties(family='Gill Sans', fname='/Library/Fonts/GillSans.ttc')
    f, ax = plt.subplots()
    plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    #plt.plot(klist,naive_rdp, 'g.-.', linewidth=2)
    plt.plot(klist,eps_rdp, 'm.-.', linewidth=2)
    plt.plot(klist, eps_optimal_rdp, 'y.-', linewidth=2)
    plt.plot(klist, eps_phi_left, 'cx--', linewidth=2)
    plt.plot(klist, eps_phi_right, 'rx--', linewidth=2)
    plt.plot(klist, eps_quarture, 'bx--', linewidth=2)
    plt.plot(klist, fft_lower, 'gs-',linewidth=1)
    plt.plot(klist, fft_higher, 'rs-', linewidth=1)
    plt.ylim([1e-12, 1e-1])
    plt.yscale('log')
    #plt.plot(klist, doc['20']['rdp'], '--k', linewidth=2)
    #plt.plot(klist, doc['20']['eps'], '--r^', linewidth=2)
    plt.legend(
        [r'BBGHS_RDP_conversion','Optimal_RDP_Conversion','$\phi$-function_lower', '$\phi$-function_upper', 'Double quadrature method','FA lower bound', 'FA higher bound'], loc='best', fontsize=17)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'Number of Composition ($\epsilon=1.0, \sigma=2$, sample probability=$0.02$)', fontsize=20)
    plt.ylabel(r'$\delta$', fontsize=20)
    ax.set_title('Title', fontproperties=props)
    #plt.show()
    plt.savefig('exp4_delta.pdf', bbox_inches='tight')

exp4_subsample_fixed_eps()



def exp5_subsample_fixed_eps():
    """
    Evaluate poisson subsample Gaussian mechanism
    sample probability = 0.02, fix delta to 1e-5 and compare epsilon over composition
    x axis is # composition
    y axis is epsilon(delta)

    Evaluate four methods
    BBGHS_RDP :eps_rdp
    Our phi-function lower bound: eps_phi_left
    Our phi-function upper bound:  eps_phi_right
    Double quadrature:eps_quadrature
    """

    delta = 1e-5

    import pickle
    prob = 0.02
    klist = [100* i for i in range(2,16)]

    doc = {}
    exp4_path = 'exp5.pkl'
    if os.path.exists(exp4_path):
        with open(exp4_path, 'rb') as f:
            doc = pickle.load(f)
            klist = klist[:]
            eps_phi_left = doc['phi_left']
            eps_phi_right = doc['phi_right']
            eps_rdp = doc['rdp']
            eps_quarture = doc['quarture']
    else:

        for sigma in [2]:
            eps_rdp = []
            eps_phi_left = []
            eps_phi_right = []
            eps_quarture = []
            for coeff in klist:
                gm1 = ExactGaussianMechanism(sigma, name='GM1')
                compose = Composition()
                poisson_sample = AmplificationBySampling(PoissonSampling=True)
                composed_mech = compose([poisson_sample(gm1, prob, improved_bound_flag=True)], [coeff])
                #phi_subsample_left = SubSampleGaussian(sigma, prob,coeff, CDF_off=False, lower_bound = True)
                #phi_subsample_right = SubSampleGaussian(sigma, prob,coeff, CDF_off=False, upper_bound = True)
                phi_quarture =  SubSampleGaussian(sigma, prob,coeff, CDF_off=False)
                eps_rdp.append(composed_mech.approxDP(delta))
                eps_quarture.append(phi_quarture.get_approxDP(delta))
                #eps_phi_left.append(phi_subsample_left.approxDP(delta))
                #eps_phi_right.append(phi_subsample_right.approxDP(delta))
                print('eps using double quarture', eps_quarture)
                #print('eps using phi lower bound', eps_phi_left)
                #print('eps using phi right bound', eps_phi_right)
                print('eps using the optimal rdp conversion', eps_rdp)
            cur_result = {}
            cur_result['rdp'] = eps_rdp
            cur_result['phi_left'] = eps_phi_left
            cur_result['phi_right'] = eps_phi_right
            cur_result['quarture'] = eps_quarture

            with open(exp4_path, 'wb') as f:
                pickle.dump(cur_result, f)


    for sigma in [2]:
        naive_rdp = []
        eps_optimal_rdp = []

        for coeff in klist:
            gm1 = ExactGaussianMechanism(sigma, name='GM1')
            compose = Composition()
            poisson_sample = AmplificationBySampling(PoissonSampling=True)
            composed_mech = compose([poisson_sample(gm1, prob, improved_bound_flag=True)], [coeff])
            eps_optimal_rdp.append(composed_mech.approxDP(delta))

    print('RDP optimal conversion', eps_optimal_rdp)
    print('quarture result', eps_quarture)

    # copy the results from fourier paper and our previous experimental results
    eps_quarture =  [0.5732509555915974, 0.7058115513738935, 0.8194825434078763, 0.9209338664859984, 1.0136435019043732, 1.099702697374365, 1.1804867199047775, 1.2569025991868228, 1.3296691895163435, 1.3993259169003278, 1.4662601850161303, 1.5308113615711694, 1.5932321716160407, 1.6637669337344212]
    eps_phi_left =  [0.5553297955867375, 0.6832218356617155, 0.7928056642133938, 0.8906191704210357, 0.9799696230766813, 1.0628838022920961, 1.1406664339350132, 1.2142613396519286, 1.2843523223848659, 1.3513924639580124, 1.4158316082377795, 1.4779534607841263, 1.5380153993392631, 1.5962516845946515]
    eps_phi_right = [0.5866220805156243, 0.7223973520962709, 0.8387942691051304, 0.9426878979377773, 1.0376345860832061, 1.1257547173351041, 1.2084450581608812, 1.2867150182217082, 1.3612194904959256, 1.4325246778969292, 1.5010566850456206, 1.567135674028485, 1.6310457295344567, 1.6929980332709857]
    fft_lower = [0.5738987073827246,0.7070330939276875,0.8213662927173179,0.92358900507979, 1.0171625920742302,1.1041565229625558,1.1859259173798518, 1.2634158671679698, 1.3373168083154394, 1.4081514706603082, 1.4763269642671626, 1.54216775460533, 1.605937390055278,1.667853467197358]
    fft_higher = [0.5738987073827246,0.7070330939276875, 0.8213662927173179, 0.92358900507979, 1.0171625920742302, 1.1041565229625558, 1.1859259173798518,1.2634158671679698,1.3373168083154394, 1.4081514706603082, 1.4763269642671626, 1.54216775460533, 1.605937390055278, 1.667853467197358]
    import matplotlib.pyplot as plt
    #epsusingphi[1.7183697391746319e-06, 1.735096619083328e-06, 2.090003729609751e-06]
    #epsusingrdp[1.1702248059464182e-09, 4.4203574134371593e-10, 3.824438543631459e-10]
    props = fm.FontProperties(family='Gill Sans', fname='/Library/Fonts/GillSans.ttc')
    f, ax = plt.subplots()
    plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    #plt.plot(klist,naive_rdp, 'g.-.', linewidth=2)
    plt.plot(klist,eps_rdp, 'm.-.', linewidth=2)
    #plt.plot(klist, eps_optimal_rdp, 'y.-', linewidth=2)
    plt.plot(klist, eps_phi_left, 'cx-', linewidth=2)
    plt.plot(klist, eps_phi_right, 'rx-', linewidth=2)
    plt.plot(klist, eps_quarture, 'y.-', linewidth=2)
    plt.plot(klist, fft_lower, 'gs--',linewidth=1)
    plt.plot(klist, fft_higher, 'rs--', linewidth=1)

    #plt.plot(klist, doc['20']['rdp'], '--k', linewidth=2)
    #plt.plot(klist, doc['20']['eps'], '--r^', linewidth=2)
    plt.legend(
        [r'BBGHS_RDP','our AFA lower bound', 'our AFA higher bound', 'Double quadrature','FA lower bound', 'FA higher bound'], loc='best', fontsize=17)
    plt.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel(r'Number of Compositions $k$', fontsize=22)
    plt.ylabel(r'$\epsilon$', fontsize=22)
    ax.set_title('Title', fontproperties=props)
    #plt.show()
    plt.savefig('exp4_eps.pdf', bbox_inches='tight')
exp5_subsample_fixed_eps()
