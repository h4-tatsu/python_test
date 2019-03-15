#%%
var="test"
print(var)


#%%
import numpy as np
import matplotlib.pyplot as plt

# 学習データを読み込む
train = np.loadtxt('test_data/data3.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]


# 標準化した学習データをプロット
plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == 0, 0], train_x[train_y == 0, 1], 'x')
plt.show()

#%%
# パラメータを初期化
theta = np.random.rand(4)

# 標準化
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)
def standardize(x):
    return(x - mu) / sigma

train_z = standardize(train_x)

# x0とx3を加える
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    x3 = x[:, 0, np.newaxis] ** 2
    return np.hstack([x0, x, x3])

X = to_matrix(train_z)

# シグモイド関数
def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

ETA = 1e-3

# 繰り返し回数
epoch = 5000
count = 0
# 学習を繰り返す
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    # ログの出力
    count += 1
    print('{}回目： theta = {}'.format(count, theta))

x1 = np.linspace(-2, 2, 100)
x2 = -(theta[0] + theta[1] * x1 + theta[3] * x1 ** 2) / theta[2]
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x1, x2, linestyle = 'dashed')
plt.show()

#%%
def classify(class_x):
    return (f(class_x) >= 0.5).astype(np.int)


theta = np.random.rand(4)

# 精度の履歴
accuracies = []

# 学習を繰り返す
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    # 現在の精度を計算
    result = classify(X) == train_y
    accuracy = len(result[result == True]) / len(result)
    accuracies.append(accuracy)

# 精度をプロット
x = np.arange(len(accuracies))

plt.plot(x, accuracies)
plt.show()