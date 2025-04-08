import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. 定义 ReLU (Arc-cosine) 核函数
# ---------------------------
def relu_kernel(X1, X2, sigma_w=1.0, sigma_b=0.0):
    """
    计算 ReLU 激活对应的 1阶 Arc-cosine 核函数，
    公式为：
      k(x, x') = sigma_w^2 * ||x|| ||x'|| * (1/π) * [sin(θ) + (π - θ)cos(θ)] + sigma_b^2,
    其中 θ = arccos((x.T x')/(||x|| ||x'||)).
    
    Parameters:
      X1: numpy 数组, shape = (N, d)
      X2: numpy 数组, shape = (M, d)
      sigma_w: 权重尺度参数, 默认1.0
      sigma_b: 偏置尺度参数, 默认0.0
      
    Returns:
      K: 核矩阵, shape = (N, M)
    """
    dot_prod = np.dot(X1, X2.T)
    norm1 = np.linalg.norm(X1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(X2, axis=1, keepdims=True)
    norm_prod = np.dot(norm1, norm2.T)
    
    eps = 1e-6  # 防止除零
    cos_theta = np.clip(dot_prod / (norm_prod + eps), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    K = sigma_w**2 * norm_prod * (1.0/np.pi) * (np.sin(theta) + (np.pi - theta)*cos_theta) + sigma_b**2
    return K

# ---------------------------
# 2. 定义 GP 回归函数（单输出）
# ---------------------------
def gp_predict(X_train, y_train, X_test, kernel_func, sigma_n=0.1):
    """
    利用 GP 的闭式公式计算预测均值和标准差.
    
    Parameters:
      X_train: 训练输入, shape = (N, d)
      y_train: 训练输出, shape = (N,)
      X_test: 测试输入, shape = (M, d)
      kernel_func: 核函数，形式 kernel_func(X1, X2)
      sigma_n: 观测噪声标准差
      
    Returns:
      mu_star: 预测均值, shape = (M,)
      sigma_star: 预测标准差, shape = (M,)
    """
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
# 3. 定义示例函数以生成 V 和 dV
# ---------------------------
def V_function(x):
    """
    模拟 V(x) 的输出（例如：V = sin(x0) + cos(x1)）
    """
    return np.sin(x[0]) + np.cos(x[1])

def dV_function(x):
    """
    模拟 dV(x) 的输出（例如：dV = cos(x0) - sin(x1)）
    """
    return np.cos(x[0]) - np.sin(x[1])

# 给定 lambda
lambda_val = 10.0

# 构造目标 F(x) = dV(x) + lambda * V(x)
# ---------------------------
# 4. 生成训练数据
# ---------------------------
np.random.seed(42)
N_train = 50
d = 2
X_train = np.random.uniform(-5, 5, size=(N_train, d))

# 针对每个训练样本，计算 V, dV, F = dV + lambda * V
F_train = np.array([dV_function(x) + lambda_val * V_function(x) for x in X_train])
# 加入噪声
noise_std = 0.1
F_train_noisy = F_train + noise_std * np.random.randn(N_train)

# ---------------------------
# 5. 生成测试数据
# ---------------------------
N_test = 100
X_test = np.random.uniform(-5, 5, size=(N_test, d))

# ---------------------------
# 6. 利用 GP 回归对 F = dV + lambda * V 进行预测
# ---------------------------
mu_F, sigma_F = gp_predict(X_train, F_train_noisy, X_test, relu_kernel, sigma_n=noise_std)

# ---------------------------
# 7. 构造置信区间 (Bound)
# ---------------------------
beta = 2.0  # 95% 置信区间近似取 2 倍标准差
lower_bound = mu_F - beta * sigma_F
upper_bound = mu_F + beta * sigma_F

# ---------------------------
# 8. 输出部分结果
# ---------------------------
print("GP Prediction for F = dV + lambda * V (first 5 test points):")
for i in range(5):
    print(f"  x = {X_test[i]}: mu = {mu_F[i]:.3f}, sigma = {sigma_F[i]:.3f}, Bound = [{lower_bound[i]:.3f}, {upper_bound[i]:.3f}]")

# ---------------------------
# 9. 绘图：展示预测均值与置信区间
# ---------------------------
plt.figure(figsize=(10,5))
plt.errorbar(np.arange(N_test), mu_F, yerr=beta*sigma_F, fmt='o', capsize=3, label="Predicted mu with 95% CI")
plt.title("GP Prediction for F = dV + lambda * V")
plt.xlabel("Test sample index")
plt.ylabel("F (dV + lambda*V)")
plt.legend()
plt.show()
