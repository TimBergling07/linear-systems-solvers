import matplotlib.pyplot as plt
import numpy as np

# 数据
n = np.array([50, 100, 200, 300])
gauss = np.array([3.51, 28.25, 170.35, 420.12])
qr    = np.array([5.52, 20.64, 69.09, 187.22])
svd   = np.array([23.64, 152.17, 601.90, 2192.34])

plt.figure(figsize=(8, 5))
plt.loglog(n, gauss, 'o-', label='Gaussian Elimination')
plt.loglog(n, qr,    's-', label='QR Decomposition')
plt.loglog(n, svd,   '^-', label='SVD Decomposition')

plt.xlabel('Matrix Size $n$')
plt.ylabel('Time (ms)')
plt.title('Chapter 5: Performance of Orthogonal Decomposition Methods')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('chapter5_performance.png', dpi=300)
plt.show()