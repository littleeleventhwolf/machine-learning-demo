# 导入numpy工具包
import numpy as np
# 使用listdir模块，用于访问本地文件
from os import listdir
from sklearn import neighbors

def img2vector(fileName):
	"""
	将加载的32*32的图片矩阵展开成一列向量
	"""
	retMat = np.zeros([1024], int) # 定义返回的矩阵，大小为1*1024
	fr = open(fileName) # 打开包含32*32大小的数字文件
	lines = fr.readlines() # 读取文件的所有行
	for i in range(32): # 遍历文件所有行
		for j in range(32): # 并将01数字存放在retMat中
			retMat[i*32+j] = lines[i][j]
	return retMat

def readDataSet(path):
	"""
	加载训练数据，并将样本标签转化为one-hot向量
	"""
	fileList = listdir(path) # 获取文件夹下的所有文件
	numFiles = len(fileList) # 统计需要读取的文件的数目
	dataSet = np.zeros([numFiles, 1024], int) # 用于存放所有的数字文件
	hwLabels = np.zeros([numFiles]) # 用于存放对应的标签（与神经网络不同）
	for i in range(numFiles): # 遍历所有的文件
		filePath = fileList[i] # 获取文件名称/路径
		digit = int(filePath.split('_')[0]) # 通过文件名获取标签
		hwLabels[i] = digit # 将对应的one-hot标签置1
		dataSet[i] = img2vector(path + '/' + filePath) # 读取文件内容
	return dataSet, hwLabels

# 调用readDataSet和img2vector函数加载数据
# 将训练的图片存放在train_dataSet中
# 将对应的标签存放在train_hwLabels中
train_dataSet, train_hwLabels = readDataSet('trainingDigits')

# 构建kNN分类器：设置查找算法以及邻居点数量(k)值
# 构建kNN的同时调用fit函数
knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
knn.fit(train_dataSet, train_hwLabels)

# 使用测试集进行评价
# 加载测试集
dataSet, hwLabels = readDataSet('testDigits')
# 使用构建好的kNN分类器对测试集进行预测，并计算预测的错误率
res = knn.predict(dataSet) # 对测试集进行预测
error_num = np.sum(res != hwLabels) # 统计分类错误的数目
num = len(dataSet) # 测试集的数目
print("Total num:", num, "Wrong num:", \
	error_num, "Wrong rate:", error_num / float(num))