# 导入numpy工具包
import numpy as np
# 使用listdir模块，用于访问本地文件
from os import listdir
from sklearn.neural_network import MLPClassifier

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
	hwLabels = np.zeros([numFiles, 10]) # 用于存放对应的标签one-hot
	for i in range(numFiles): # 遍历所有的文件
		filePath = fileList[i] # 获取文件名称/路径
		digit = int(filePath.split('_')[0]) # 通过文件名获取标签
		hwLabels[i][digit] = 1.0 # 将对应的one-hot标签置1
		dataSet[i] = img2vector(path + '/' + filePath) # 读取文件内容
	return dataSet, hwLabels

# 调用readDataSet和img2vector函数加载数据
# 将训练的图片存放在train_dataSet中
# 将对应的标签存放在train_hwLabels中
train_dataSet, train_hwLabels = readDataSet('trainingDigits')

#############################
#       构建神经网络        #
# 1.设置网络的隐藏层数      #
# 2.设置各隐藏层神经元个数  #
# 3.设置激活函数            #
# 4.设置学习率              #
# 5.设置优化方法            #
# 6.设置最大迭代次数        #
#############################
clf = MLPClassifier(hidden_layer_sizes=(100,),
	activation='logistic', solver='adam',
	learning_rate_init=0.0001, max_iter=2000)
# 使用训练数据训练构建好的神经网络
# fit函数能够根据训练集及对应标签集自动设置多层感知机的输入与输出层的神经元个数
clf.fit(train_dataSet, train_hwLabels)

# 使用测试集进行评价
# 加载测试集
dataSet, hwLabels = readDataSet('testDigits')
# 使用训练好的MLP对测试集进行预测，并计算错误率
res = clf.predice(dataSet) # 对测试集进行预测
error_num = 0 # 统计预测错误的数目
num = len(dataSet) # 测试集的数目
for i in range(num): # 遍历预测结果
	# 比较长度为10的数组，返回包含0和1的数组，0为不同，1为相同
	# 若预测结果与真实结果相同，则10个数字全为1，否则不全为1
	if np.sum(res[i] == hwLabels[i]) < 10:
		error_num += 1
print("Total num:", num, "Wrong num:", \
	error_num, "Wrong rate:", error_num / float(num))