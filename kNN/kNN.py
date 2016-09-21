from os import listdir
from numpy import *
import operator

# create data set with 4 points
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# k-Nearest Neighbor
def classify0(inX, dataSet, labels, k):
    """
    For every point in our dataset:
        calculate the distance between inX and the current point
        sort the distances in increasing order
        take k items with lowest distances to inX
        find the majority class among these items
        return the majority class as our prediction for the class of inX

    params:
        inX: the input vector to classify
        dataSet: our full matrix of training examples
        labels: a vector of labels
        k: the number of nearest neighbors to use in the voting
    """
    # Distance calculation
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # Voting with lowest k distances
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sort dictionary
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]

# text record to numpy parsing code
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    # get number of lines in file
    numberOfLines = len(arrayOLines)
    # create numpy matrix to return
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # parse line to a list
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# data-normalizing code
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    # Element-wise division
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# classifier testing code for dating site
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], \
                                     datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" \
              % (classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is %f" % (errorCount/float(numTestVecs))

# dating site predictor function
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input( \
                        "percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - \
                                 minVals)/ranges, normMat, datingLabels, 3)
    print "you will probably like this person: ", \
          resultList[classifierResult - 1]

# converting images into test vectors
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

# handwritten digits testing code
def handwritingClassTest():
    hwLabels = []
    # get contents of directory
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # process class num from filename
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, \
                                     trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" \
              % (classifierResult, classNumStr)
        if(classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

