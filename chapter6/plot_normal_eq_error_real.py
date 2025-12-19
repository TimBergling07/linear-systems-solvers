import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('sigma_vs_error_fixed.csv', delimiter=',', skiprows=1)
sigma_min = data[:, 0]
error = data[:, 1]

plt.figure(figsize=(8, 5))
plt.loglog(sigma_min, error, 'bo-', markersize=5)
plt.xlabel(r'Smallest Singular Value $\sigma_{\min}$')
plt.ylabel('Solution Error (Normal Equations)')
plt.title('Error Grows as $1/\\sigma_{\\min}^2$ in Ill-Conditioned Systems')
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.savefig('normal_eq_error_real.png', dpi=300)
plt.show()