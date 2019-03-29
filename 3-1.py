#%%
var="test"
print(var)


#%%
import numpy as np
import matplotlib.pyplot as plt

# 学習データを読み込む
train = np.loadtxt('test_data/images1.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# プロット
# plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
# plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
# plt.axis('scaled')
# plt.show()
print(train_x[train_y == 1, 0])
print(train_x[train_y == 1, 1])
print(train_x[train_y == -1, 0])
print(train_x[train_y == -1, 0])

plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
plt.axis('scaled')
plt.show()

# 重みの初期化
w = np.random.rand(2)

# 識別関数
def f(x):
    if np.dot(w, x) >= 0:
        return 1
    else:
        return -1

# 繰り返し回数
epoch = 10

# 更新回数
count = 0
count_1 = 0
count_2 = 0

# 重みを学習する
for _ in range(epoch):

    print('count 1 = {}'.format(count_1))
    count_1 += 1
    for x, y in zip(train_x, train_y):
        print('count 2 = {}'.format(count_2))
        count_2 += 1
        if f(x) != y:
            w = w + y * x
            # ログの出力
            count += 1
            print('{}回目： w = {}'.format(count, w))

x1 = np.arange(0, 500)
print('w = {}'.format(w))
plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
plt.plot(x1, -w[0] / w[1] * x1, linestyle = 'dashed')
plt.show()

#%%
f([200, 100])

#%%
f([100, 200])

#%%
f([100, 100])

#%%
f([99, 100])

#%%
f([99.5, 100])

#%%
f([99.6, 100])


# #%%
# plt.style.use('dark_background')

# fig, ax = plt.subplots()

# L = 6
# x = np.linspace(0, L)
# ncolors = len(plt.rcParams['axes.prop_cycle'])
# shift = np.linspace(0, L, ncolors, endpoint=False)
# for s in shift:
#     ax.plot(x, np.sin(x + s), 'o-')
# ax.set_xlabel('x-axis')
# ax.set_ylabel('y-axis')
# ax.set_title("'dark_background' style sheet")

# plt.show()

#%%
