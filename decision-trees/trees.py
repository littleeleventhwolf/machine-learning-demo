from math import log

# function to calculate the Shannon entropy of dataset
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # create dictionary of all possible classes
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # logarithm base 2
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# simple data about fish identification
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# dataset splitting on a given feature
def splitDataSet(dataSet, axis, value):
    """
    params:
        dataSet: the dataset we will split
        axis: the feature we will split on
        value: the value of the feature to return
    """
    # create separate list
    retDataSet = []
    # cut out the feature split on
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


