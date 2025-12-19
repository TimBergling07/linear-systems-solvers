import numpy as np

def compute_error_and_sigma(eps):
    # Define true solution
    x_true = np.array([1.0, 2.0])
    
    # Build A
    A = np.array([[1.0, 1.0],
                  [1.0, 1.0 + eps]], dtype=np.float64)
    
    # Compute b = A @ x_true (so x_true is the exact least-squares solution)
    b = A @ x_true

    # Smallest singular value
    s = np.linalg.svd(A, compute_uv=False)
    sigma_min = s[-1]

    # Normal equations
    try:
        AtA = A.T @ A
        Atb = A.T @ b
        x_ne = np.linalg.solve(AtA, Atb)
        error = np.linalg.norm(x_ne - x_true)
    except np.linalg.LinAlgError:
        error = np.nan

    return sigma_min, error

# Sweep eps from 1e-1 to 1e-12 (avoid too small where everything breaks)
epsilons = np.logspace(-1, -12, 30)
sigmas = []
errors = []

for eps in epsilons:
    s_min, err = compute_error_and_sigma(eps)
    sigmas.append(s_min)
    errors.append(err)

# Save to CSV
data = np.column_stack((sigmas, errors))
np.savetxt('sigma_vs_error_fixed.csv', data, delimiter=',', header='sigma_min,error', comments='')
print("✅ Data saved to 'sigma_vs_error_fixed.csv'")
print("\nFirst few rows:")
for i in range(5):
    print(f"sigma_min = {sigmas[i]:.3e}, error = {errors[i]:.3e}")