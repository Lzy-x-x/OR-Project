import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def sig(z):
    return 1 / (1 + np.exp(-z))

def c(z, A, b):
    return b * (A @ z)

def obj(z, A, b, m0):
    return m0 * (np.sum(np.log(1 + np.exp(-c(z, A, b)))) + (1 / 100) * np.dot(z.T, z))

def grad(z, A, b, m0):
    return m0 * (-A.T @ (b * (1 - sig(c(z, A, b)))) + (1 / 50) * z)

def hessian(z, A, b, m0, n):
    Sig = sig(c(z, A, b))
    Vec = Sig * (1 - Sig)
    Mat = np.tile(Vec, (n, 1)).T
    return m0 * (Mat * (A.T @ A) + (1 / 50) * np.eye(n))

def backtracking(z, A, b, d, alpha0, m0):
    gamma = 0.1
    c1 = 0.1
    t = 0
    alpha = alpha0 * gamma ** t
    while obj(z + alpha0 * gamma ** t * d, A, b, m0) > obj(z, A, b, m0) + c1 * alpha0 * gamma ** t * (
            grad(z, A, b, m0).T @ d):
        alpha = alpha0 * gamma ** t
        t += 1
    return alpha, t

def newton(z, A, b, epsilon, m0, n):
    values = np.zeros((1001, 3))
    i = 0
    alpha0 = 1
    object_val = obj(z, A, b, m0)
    values[i, 0] = object_val
    gradient = grad(z, A, b, m0)
    Hessian = hessian(z, A, b, m0, n)
    d = -np.linalg.solve(Hessian, gradient)
    alpha, t = backtracking(z, A, b, d, alpha0, m0)
    values[i, 2] = t
    z = z + alpha * d
    values[i, 1] = np.linalg.norm(gradient)

    while values[i, 1] >= epsilon:
        i += 1
        object_val = obj(z, A, b, m0)
        values[i, 0] = object_val
        gradient = grad(z, A, b, m0)
        Hessian = hessian(z, A, b, m0, n)
        d = -np.linalg.solve(Hessian, gradient)
        alpha, t = backtracking(z, A, b, d, alpha0, m0)
        values[i, 2] = t
        z = z + alpha * d
        values[i, 1] = np.linalg.norm(gradient)
        print(f"第{i + 1}次迭代")

    return values, i

# 读取数据 (假设是.mat文件)
# 使用scipy.io加载MATLAB .mat文件
import scipy.io

data = scipy.io.loadmat('a9a.txt.mat')
A = data['data'][0][0].T
b = data['data'][0][1]

m, n = A.shape
m0 = 1 / m

z = np.zeros(n)  # 初始向量为全0向量

values, iterations = newton(z, A, b, 1e-6, m0, n)

# 打印结果
print(f'结果为 {values[iterations, 0]} 迭代次数{iterations}')

# 绘图
plt.figure(figsize=(10, 8))

# 函数值与最优值的差收敛速度
plt.subplot(3, 1, 1)
plt.plot(np.arange(iterations + 1), values[:iterations + 1, 0] - values[iterations, 0])
plt.title('函数值与最优值的差收敛速度')
plt.xlabel('迭代次数')
plt.ylabel('函数值与最优值的差')

# 梯度范数收敛速度
plt.subplot(3, 1, 2)
plt.plot(np.arange(iterations + 1), values[:iterations + 1, 1])
plt.title('梯度范数收敛速度')
plt.xlabel('迭代次数')
plt.ylabel('梯度范数')

# 线搜索收敛次数
plt.subplot(3, 1, 3)
plt.plot(np.arange(iterations + 1), values[:iterations + 1, 2])
plt.title('线搜索收敛次数')
plt.xlabel('迭代次数')
plt.ylabel('线搜索收敛次数')

plt.tight_layout()
plt.show()