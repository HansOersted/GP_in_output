import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1.Define RBF (Gaussian) Kernel
# ---------------------------
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    Gaussian Kernel (RBF):
      k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * length_scale^2))
    """
    sqdist = np.sum(X1**2, axis=1).reshape(-1,1) + \
             np.sum(X2**2, axis=1) - 2*np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# ---------------------------
# 2. Define GP Predictor
# ---------------------------
def gp_predict(X_train, y_train, X_test, kernel_func, sigma_n=0.1):
    K = kernel_func(X_train, X_train)
    N = X_train.shape[0]
    K += sigma_n**2 * np.eye(N)

    K_star = kernel_func(X_test, X_train)  # shape (M, N)
    K_star_star = kernel_func(X_test, X_test)
    diag_K_star_star = np.diag(K_star_star)

    alpha = np.linalg.solve(K, y_train)
    mu_star = np.dot(K_star, alpha)

    v = np.linalg.solve(K, K_star.T)
    var_star = diag_K_star_star - np.sum(K_star * v.T, axis=1)
    sigma_star = np.sqrt(np.maximum(var_star, 0))

    return mu_star, sigma_star

# ---------------------------
# 3. Assign V(x) and dV(x)
# ---------------------------
def V_function(x):
    return np.sin(x[0]) + np.cos(x[1])

def dV_function(x):
    return np.cos(x[0]) - np.sin(x[1])

# ---------------------------
# 4. Construct F(x) = dV(x) + lambda * V(x)
# ---------------------------
lambda_val = 10.0
np.random.seed(42)
N_train = 50
d = 2
X_train = np.random.uniform(-5, 5, size=(N_train, d))

F_train = np.array([dV_function(x) + lambda_val * V_function(x) for x in X_train])
noise_std = 0.1
F_train_noisy = F_train + noise_std * np.random.randn(N_train)

# ---------------------------
# 5. Generate test data
# ---------------------------
N_test = 1
X_test = np.random.uniform(-5, 5, size=(N_test, d))

# ---------------------------
# 6. Use GP to predict F
# ---------------------------
mu_F, sigma_F = gp_predict(
    X_train, F_train_noisy, X_test,
    lambda X1, X2: rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0),
    sigma_n=noise_std
)

# ---------------------------
# 7. Calculate confidence intervals
# ---------------------------
beta = 2.0
lower_bound = mu_F - beta * sigma_F
upper_bound = mu_F + beta * sigma_F

# ---------------------------
# 8. Print prediction results
# ---------------------------
print("GP Prediction using RBF Kernel:")
for i in range(N_test):
    print(f"x = {X_test[i]}: mu = {mu_F[i]:.3f}, sigma = {sigma_F[i]:.3f}, Bound = [{lower_bound[i]:.3f}, {upper_bound[i]:.3f}]")

# ---------------------------
# 9. Plot results
# ---------------------------
plt.figure(figsize=(10,5))
plt.errorbar(np.arange(N_test), mu_F, yerr=beta*sigma_F, fmt='o', capsize=3, label="Predicted mu with 95% CI")
plt.title("GP Prediction for F = dV + lambda * V using RBF Kernel")
plt.xlabel("Test sample index")
plt.ylabel("F (dV + lambda*V)")
plt.legend()
plt.grid(True)
plt.show()
