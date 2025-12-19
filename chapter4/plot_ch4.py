import matplotlib.pyplot as plt
import numpy as np

# 数据
n = np.array([10, 20, 50, 100])
gauss = np.array([0.1, 0.55, 3.50, 27.09])      # 0.1 代替 0.00
inv   = np.array([0.1, 0.1,  4.01, 28.06])
lu    = np.array([0.1, 0.54, 1.50, 6.61])

plt.figure(figsize=(8, 5))
plt.loglog(n, gauss, 'o-', label='Gaussian Elimination')
plt.loglog(n, inv,   's-', label='Matrix Inversion')
plt.loglog(n, lu,    '^-', label='LU Decomposition')

plt.xlabel('Matrix Size $n$')
plt.ylabel('Time (ms)')
plt.title('Chapter 4: Performance Comparison of Linear Solvers')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('chapter4_performance.png', dpi=300)
plt.show()