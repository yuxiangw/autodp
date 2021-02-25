import math
from autodp.calibrator_zoo import eps_delta_calibrator,generalized_eps_delta_calibrator, ana_gaussian_calibrator
from autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism,SubsampleGaussianMechanism, GaussianMechanism, ComposedGaussianMechanism, LaplaceMechanism
from autodp.transformer_zoo import Composition, AmplificationBySampling
from absl.testing import absltest
from absl.testing import parameterized
"""
Unit tests for  calibrating noise to privacy budgets.
Cases 1: single parameter, no subsample or composition

"""
def testcase_single_para():

	test_case = []
	# each test_case = {'noise_para','delta','eps'} range sigma from 1 to 100
	sigma_list = [int(1.6**i) for i in range(int(math.floor(math.log(100,1.6)))+1)]
	for sigma in sigma_list:
		for delta in [  1e-3, 1e-4,1e-5,1e-6]:
			gm = ExactGaussianMechanism(sigma, name = 'GM')
			cur_test = {'sigma':sigma, 'delta':delta,'eps':gm.get_approxDP(delta)}
			test_case.append(cur_test)
	return test_case

def _single_para(eps, delta,params_name='sigma', bounds=[0,100]):
	"""
	test calibrator
	"""
	calibrate = eps_delta_calibrator()
	mech1 = calibrate(ExactGaussianMechanism,eps,delta,[0,100],name='GM')
	return mech1.params[params_name]
tol = 0.01
class TestGaussianMechanism(parameterized.TestCase):
	params = testcase_single_para()
	@parameterized.parameters(p for p in params)
	def test_single_para(self,eps,delta, sigma):
		autodp_value = _single_para(eps, delta, params_name='sigma')
		self.assertGreaterEqual(autodp_value, (1.-tol)*sigma)


"""
Unit tests for  calibrating noise to privacy budgets.
Cases 2: multi-parameter 

"""
def testcase_multi_para():
	"""
	create test cases when there are multi parameters (e.g., #coeff and sigma in Composed Gaussian mechanism)
	"""

	test_case = []
	# each test_case = {'noise_para','delta','coeff','eps'} range sigma from 1 to 100, coeff is the number of composition
	sigma_list = [int(1.6**i) for i in range(int(math.floor(math.log(100,1.6)))+1)]
	for sigma in sigma_list:
		for coeff in [1, 10, 100, 1000]:
			for delta in [  1e-3, 1e-4,1e-5, 1e-6]:
				gm = ExactGaussianMechanism(sigma, name = 'GM')
				compose = Composition()
				composed_mech = compose([gm], [coeff])
				cur_test = {'sigma':sigma, 'delta':delta,'coeff':coeff, 'eps':composed_mech.get_approxDP(delta)}
				test_case.append(cur_test)
	return test_case
def _multi_para(eps, delta,coeff, params_name='sigma', bounds=[0,100]):
	"""
	test calibrator with multi-parameters
	"""
	general_calibrate = generalized_eps_delta_calibrator()
	params = {}
	params[params_name] = None
	params['coeff'] = coeff
	mech1 = general_calibrate(ComposedGaussianMechanism,eps,delta,[0,1000], params=params, para_name = params_name,name='GM')

	return mech1.params[params_name]
class TestGeneralizedMechanism(parameterized.TestCase):
	"""
	We test generalized calibrator that deals with compostion coeff
	"""
	params = testcase_multi_para()
	@parameterized.parameters(p for p in params)

	def test_single_para(self, eps, delta,coeff, sigma):
		autodp_value = _multi_para(eps=eps, delta=delta,coeff=coeff, params_name='sigma')
		self.assertGreaterEqual(autodp_value, (1. - tol) * sigma)


