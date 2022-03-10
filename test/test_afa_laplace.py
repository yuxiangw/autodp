from autodp.mechanism_zoo import LaplaceMechanism
from autodp.transformer_zoo import Composition, ComposeAFA, ComposeGaussian, AmplificationBySampling
import matplotlib.pyplot as plt


eps = 0.01
delta = 1e-6
laplace_phi = LaplaceMechanism( b=1./eps, phi_off=False)
laplace_rdp = LaplaceMechanism(b=1./eps)


compose_afa = ComposeAFA()
compose_rdp = Composition()
eps_rdp = []
eps_phi = []


klist = [int(100*i) for i in range(2,10)]
print(klist)
for k in klist:
    print('k', k)
    cur_composed_phi = compose_afa([laplace_phi], [k])
    cur_composed_rdp = compose_rdp([laplace_rdp], [k])
    eps_rdp.append(cur_composed_rdp.approxDP(delta))
    eps_phi.append(cur_composed_phi.approxDP(delta))
    #eps_phi.append(cur_composed_phi.approx_delta(delta))



print('eps_rdp', eps_rdp)
print('eps_phi', eps_phi)

plt.figure(figsize = (6,6))

plt.plot(klist, eps_rdp , 'm', linewidth=2)
plt.plot(klist, eps_phi, 'D--', color = 'pink', linewidth=2)

plt.yscale('log')
plt.legend(
    [r'RDP','Our AFA'], loc='best', fontsize=18)
plt.grid(True)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r'Number of Compositions $k$', fontsize=18)
plt.ylabel(r'$\epsilon$', fontsize=18)
plt.title(r'Laplace mechanism, $b=0.5$')
plt.show()
