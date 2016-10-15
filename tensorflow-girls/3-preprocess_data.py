from __future__ import print_function
from __future__ import division

from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np

def reformat(samples, labels):
	"""
	change the shape of initial data:
	(image_height, image_width, channels, image_num)
	 ==> 
	(image_num, image_height, image_width, channels)
	and change labels to one-hot encoding
	"""
	samples = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)

	labels = np.array([x[0] for x in labels]) # slow code whatever
	one_hot_labels = []
	for num in labels:
		one_hot = [0.0] * 10
		if num == 10:
			one_hot[0] = 1.0
		else:
			one_hot[num] = 1.0
		one_hot_labels.append(one_hot)
	labels = np.array(one_hot_labels).astype(np.float32)
	return samples, labels

def normalize(samples):
	"""
	map 0~255 to -1.0~+1.0
	gray: 3 channels ==> 1 channel 
	"""
	# gray
	a = np.add.reduce(samples, keepdims=True, axis=3)
	a = a/3.0
	# normalize
	return a/128.0 -1.0

def distribution(labels, name):
	"""
	look for the labels distribution
	"""
	count = {}
	for label in labels:
		key = 0 if label[0] == 10 else label[0]
		if key in count:
			count[key] += 1
		else:
			count[key] = 1
	x = []
	y = []
	for k, v in count.items():
		# print(k, v)
		x.append(k)
		y.append(v)

	y_pos = np.arange(len(x))
	plt.bar(y_pos, y, align='center', alpha=0.5)
	plt.xticks(y_pos, x)
	plt.ylabel('Count')
	plt.title(name + ' Label Distribution')
	plt.show()

def inspect(dataset, labels, i):
	"""
	inspect image
	"""
	print(labels[i])
	plt.imshow(dataset[i])
	plt.show()

traindata = load('SVHN/train_32x32.mat')
testdata  = load('SVHN/test_32x32.mat')
#extradata = load('SVHN/extra_32x32.mat')

print('Train Data Samples Shape: ', traindata['X'].shape)
print('Train Data Labels Shape: ', traindata['y'].shape)

print('Test Data Samples Shape: ', testdata['X'].shape)
print('Test Data Labels Shape: ', testdata['y'].shape)

#print('Extra Data Samples Shape: ', extradata['X'].shape)
#print('Extra Data Labels Shape: ', extradata['y'].shape)

train_samples = traindata['X']
train_labels = traindata['y']
test_samples = testdata['X']
test_labels = testdata['y']

_train_samples, _train_labels = reformat(train_samples, train_labels)
_test_samples, _test_labels = reformat(test_samples, test_labels)

num_labels = 10
image_size = 32

if __name__ == "__main__":
	# inspect(_train_samples, _train_labels, 1000)
	# normalize(_train_samples)
	# inspect(_train_samples, _train_labels, 1000)
	distribution(train_labels, 'Train Labels')
	distribution(test_labels, 'Test Labels')