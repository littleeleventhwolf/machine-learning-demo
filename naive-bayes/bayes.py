from numpy import *

# word list to vector function
def loadDataSet():
	postingList = [['my', 'dog', 'has', 'flea', \
	                'problems', 'help', 'please'],
	               ['maybe', 'not', 'take', 'him', \
	                'to', 'dog', 'park', 'stupid'],
	               ['my', 'dalmatian', 'is', 'so', 'cute', \
	                'I', 'love', 'him'],
	               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
	               ['mr', 'licks', 'ate', 'my', 'steak', 'how' \
	                'to', 'stop', 'him'],
	               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0, 1, 0, 1, 0, 1] # 1 is abusive, 0 not
	return postingList, classVec

def createVocabList(dataSet):
	vocabSet = set([]) # create an empty set
	for document in dataSet:
		vocabSet = vocabSet | set(document) # create the union of two sets
	return list(vocabSet)

# naive bayes set-of-words model
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList) # create a vector of all 0s
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print "the word: %s is not in my Vocabulary!" % word
	return returnVec

# naive bayes bag-of-words model
def bagOfWords2VecMN(vocabList, inputSet):
	returnVec = [0] * len(vocabList) # create a vector of all 0s
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
		#else:
		#	print "the word: %s is not in my Vocabulary!" % word
	return returnVec

# naive bayes classifier training function
def trainNBO(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	# initialize probability
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0

	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			# vector addition
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
			
	# element-wise division
	p1Vect = log(p1Num / p1Denom) # change to log()
	p0Vect = log(p0Num / p0Denom) # change to log()
	return p0Vect, p1Vect, pAbusive

# define bayes classify function
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1) # element-wise multiplication
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	listOPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	p0V, p1V, pAb = trainNBO(array(trainMat), array(listClasses))
	testEntry = ['love', 'my', 'dalmatian']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
	testEntry = ['stupid', 'garbage']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
