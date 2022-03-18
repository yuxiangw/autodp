import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from autodp.phi_bank import phi_laplace, phi_gaussian
# this file applies Fourier Accountant to compute numerical inversion of characteristic functions.
from scipy.fft import fft, ifft
from scipy.stats import norm

"""
eta is chosen to ensure that the range of F(z) (cdf at z) is fully represented.
H is chosen to define the number of points (total contains 2H-1)
the range of random variable (log(p/q)) is [-pi/eta, pi/eta]
z_j (the j-th log(p/q))

"""

H = 10**5
eta = 1e-2
lam = 2*np.pi/(eta*(2*H-1))
#lam = 2*np.pi/(eta*(2*H-1))
#b = lam*(2*H-1)/2.
b = -np.pi/eta
m_list = np.array([i for i in range(2*H-1)])
# _list contains the range of t, where t is the order of log(p/q) in the characteristic functions.
t_list = [m_hat + 1 - H for m_hat in m_list]
sigma = 5.
params = {'sigma':sigma}
#params = {'b': 100.}
n_comp = 10
#log_phi = lambda l: 10*phi_laplace(params, l)
log_phi = lambda l: 10*phi_gaussian(params, l)
def f_phi(l):
    if l==0:
        return 0
    return np.exp(log_phi(l*eta) - 1.j*eta*b*l)/(l)

# v = j - H, v ranges from []
phi_list =np.array([f_phi(m_hat) for m_hat in t_list])

fft_res = fft(phi_list)
#fft_norm = [fft_res[j]*np.exp(1.j*j*np.pi) for j in range(2*H-1)]
fft_norm = [fft_res[j]*np.exp(1.j*(H-1)*2.0/(2*H-1)*j*np.pi) for j in range(2*H-1)]
"""
cdf[z] denotes the cdf of log(p/q) when evaluates at z, where z = 2pi*(j-H)/(2eta(H-1))
the range of z is [-pi/eta, pi/eta], the mesh-size is pi/(eta*H), so we need eta*H to be a large number.
cdf[z] = 0.5 + eta*z/(2pi) -fft_res[z]
"""
cdf = np.zeros([2*H+2])

def convert_z(j):
    # given j returns z
    # z_j = -b + lam*j
    return b + lam*j

cdf = [ 0.5 +eta*convert_z(j)/(2*np.pi)- 1./(2*np.pi*1.j)*fft_norm[j] for j in range(2*H-1)]
#cdf = [ 0.5+eta*convert_z(j)/(2*np.pi) - fft_res[j] for j in range(2*H-1)]
# test when privacy loss is evaluated at test_point
#print('cdf', cdf)
cdf_map = {}
for j in range(2*H-1):
    cdf_map[convert_z(j)] = cdf[j]
test_point = convert_z(int(.9*H))
eta_std = 1./(2*sigma**2)
print('max cdf', max(cdf), min(cdf))
#print('test_point', test_point)
#print('gt cdf', norm.cdf((test_point-eta_std)/(np.sqrt(2*eta_std))), 'fft cdf', cdf_map[test_point])

