#%%
var="test"
print(var)


#%%
import numpy as np
import matplotlib.pyplot as plt

# 学習データを読み込む
train = np.loadtxt('test_data/images2.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# パラメータを初期化
theta = np.random.rand(3)

# 標準化
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)
def standardize(x):
    return(x - mu) / sigma

train_z = standardize(train_x)

# x0を加える
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    return np.hstack([x0, x])

X = to_matrix(train_z)

# 標準化した学習データをプロット
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.show()

# シグモイド関数
def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

ETA = 1e-3

# 繰り返し回数
epoch = 5000

# 学習を繰り返す
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    # ログの出力
    count += 1
    print('{}回目： w = {}'.format(count, w))

x0 = np.linspace(-2, 2, 100)
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x0, -(theta[0] + theta[1] * x0) / theta[2], linestyle = 'dashed')
plt.show()

#%%
f(to_matrix(standardize([
    [200, 100],
    [100, 200]
])))

def classify(class_x):
    return (f(class_x) >= 0.5).astype(np.int)

classify(to_matrix(standardize([
    [200, 100],
    [100, 200]
])))









# # プロット
# plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
# plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
# plt.axis('scaled')
# plt.show()

# # 重みの初期化
# w = np.random.rand(2)

# # 識別関数
# def f(x):
#     if np.dot(w, x) >= 0:
#         return 1
#     else:
#         return -1

# # 繰り返し回数
# epoch = 10

# # 更新回数
# count = 0

# # 重みを学習する
# for _ in range(epoch):
#     for x, y in zip(train_x, train_y):
#         if f(x) != y:
#             w = w + y * x
#             # ログの出力
#             count += 1
#             print('{}回目： w = {}'.format(count, w))

# x1 = np.arange(0, 500)
# plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
# plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
# plt.plot(x1, -w[0] / w[1] * x1, linestyle = 'dashed')
# plt.show()
