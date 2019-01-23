#%%
var="test"
print(var)


#%%
import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt('test_data/click.csv', delimiter=',', skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

plt.plot(train_x, train_y, 'o')
plt.show()

#%%
# パラメータの初期化
theta0 = np.random.rand()
theta1 = np.random.rand()

# 予測関数
def f(x):
    return theta0 + theta1 * x

# 目的関数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# 標準化
mu = train_x.mean()
sigma = train_x.std()

def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

plt.plot(train_z, train_y, 'o')
plt.show()

#%%
# 学習率
ETA = 1e-3

# 誤差の差分
diff = 1

# 更新回数
count = 0

# 学習を繰り返す
error = E(train_z, train_y)

while diff > 1e-2:
    # 更新結果を一時変数に保存
    tmp0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    tmp1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)
    
    # パラメータを更新
    theta0 = tmp0
    theta1 = tmp1

    # 前回の誤差と差分を計算
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error
    
    # ログの出力
    count += 1
    log = '{}回目: theta0 = {:.3f}, theta1 = {:.3f}, 差分 = {:.4f}'
    print(log.format(count, theta0, theta1, diff))


#%%
x = np.linspace(-2.5, 2.5, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(x))
plt.show()

#%%
theta = np.random.rand(3)

# 学習データの行列を作る
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2])

X = to_matrix(train_z)

# 予測関数
def f1(x):
    return np.dot(x. theta)

#%%
# 誤差の差分
diff = 1

# 学習を繰り返す
error = E(X, train_y)

while diff > 1e-2:
    # パラメータの更新
    theta = theta - ETA * np.dot(f1(X) - train_y, X)
    # 前回の誤差との差分を計算
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f1(to_matrix(x)))
plt.show()
