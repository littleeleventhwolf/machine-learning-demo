import numpy as np
# 通过sklearn.linear_model加载岭回归方法
from sklearn.linear_model import Ridge
# 加载交叉验证模块
from sklearn import cross_validation
# 加载matplotlib模块
import matplotlib.pyplot as plt
# 通过sklearn.preprocessing加载PolynomialFeatures用于创建多项式特征，如ab、a^2、b^2
from sklearn.preprocessing import PolynomialFeatures

# 使用numpy的方法从txt文件中加载数据
data = np.genfromtxt('data.txt')
# 使用plt展示车流量信息
plt.plot(data[:, 4])

# X用于保存0-3维数据，即属性
X = data[:, :4]
# y用于保存地4维的数据，即车流量
y = data[:, 4]
# 用于创建最高次数6次方的多项式特征，多次试验后决定采用6次
poly = PolynomialFeatures(6)
# X为创建的多项式特征
X = poly.fit_transform(X)

# 将所有数据划分为训练集和测试集，test_size表示测试集的比例
# random_state是随机数种子
train_set_X, test_set_X, train_set_y, test_set_y = \
	cross_validation.train_test_split(X, y, test_size=0.3, \
		random_state=0)

# 接下来我们创建岭回归实例
clf = Ridge(alpha=1.0, fit_intercept=True)
# 调用fit函数使用训练集训练回归器
clf.fit(train_set_X, train_set_y)
# 利用测试集计算回归曲线的拟合优度，clf.score返回值为0.7375
# 拟合优度，用于评价拟合好坏，最大为1，当对所有输入都输出同一个值时，拟合优度为0
clf.score(test_set_X, test_set_y)

# 接下来我们画一段200到300范围内的拟合曲线
start = 200
end = 300
# 是调用predict函数的拟合值
y_pre = clf.predict(X)
time = np.arange(start, end)
# 展示真实数据（蓝色）以及拟合曲线（红色）
plt.plot(time, y[start : end], 'b', label="real")
plt.plot(time, y_pre[start : end], 'r', label="predict")
# 设置图例的位置
plt.legend(loc='upper left')
plt.show()