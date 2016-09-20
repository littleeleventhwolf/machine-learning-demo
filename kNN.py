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

