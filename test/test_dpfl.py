import math
from autodp.calibrator_zoo import eps_delta_calibrator,generalized_eps_delta_calibrator, ana_gaussian_calibrator
from autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism,SubsampleGaussianMechanism, GaussianMechanism, ComposedGaussianMechanism, LaplaceMechanism
from autodp.transformer_zoo import Composition



general_calibrate = generalized_eps_delta_calibrator()
params = {}
params['sigma'] = None
params['coeff'] = 400
mech1 = general_calibrate(ComposedGaussianMechanism,1.0, 1e-3,[0,1000], params=params, para_name ='sigma',name='GM')
cali_sigma = mech1.params['sigma']
print('final sigma', cali_sigma)
gm = ExactGaussianMechanism(cali_sigma, name = 'GM')
compose = Composition()
composed_mech = compose([gm], [400])
print('final result', composed_mech.get_approxDP(1e-3))


