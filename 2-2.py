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
print(theta0)
print(theta1)
# 学習率
ETA = 1e-3

# 誤差の差分
diff = 1

# 更新回数
count = 0

# 学習を繰り返す
error = E(train_z, train_y)

while diff > 1e-2:
    # 更新結果を更新
    theta0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    theta1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)
    
    # 前回の誤差と差分を計算
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error
    
    # ログの出力
    count += 1
    log = '{}回目: theta0 = {:.3f}, theta1 = {:.3f}, 差分 = {:.4f}'
    print(log.format(count, theta0, theta1, diff))


#%%
x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(x))
plt.show()

#%%
print(f(standardize(100)))

print(f(standardize(200)))

print(f(standardize(300)))


#%%
a = np.random.randint(0, 10, size=(2,5))
print(a)

print(np.sum(a))

arr = np.asarray([1,2,3])

print(arr)

#%%
print(arr)
print(arr.mean())
print(arr.std())
print(arr.max())
print(arr.min())

a = np.random.randint(0, 10, size=(2,5))
print(a)
print(np.sum(a, axis=0))