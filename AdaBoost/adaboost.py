from numpy import *

def loadSimpData():
	dataMat = matrix([[1., 2.1],
		[2., 1.1],
		[1.3, 1.],
		[1., 1.],
		[2., 1.]])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return dataMat, classLabels

# decision stump-generating functions
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	"""
	classify the data
	"""
	retArray = ones((shape(dataMatrix)[0], 1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:, dimen] > threshVal] = -1.0
	return retArray