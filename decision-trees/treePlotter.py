import matplotlib.pyplot as plt

# define box and arrow formatting
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# draws annotations with arrows
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	createPlot.ax1.annotate(nodeTxt, xy=parentPt,
		xycoords='axes fraction', xytext=centerPt,
		textcoords='axes fraction', va="center",
		ha="center", bbox=nodeType, arrowprops=arrow_args)

# new version
def createPlot(inTree):
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	#plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
	#plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))
	plotTree.xOff = -0.5/plotTree.totalW
	plotTree.yOff = 1.0
	plotTree(inTree, (0.5, 1.0), '')
	plt.show()

# idetifying the number of leaves in a tree and the depth
def getNumLeafs(myTree):
	numLeafs = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		# test if node is dictionary
		if type(secondDict[key]).__name__ == 'dict':
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1
	return numLeafs

def getTreeDepth(myTree):
	maxDepth = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			thisDepth = 1 + getTreeDepth(secondDict[key])
		else:
			thisDepth = 1
		if thisDepth > maxDepth:
			maxDepth = thisDepth
	return maxDepth

# this will save you the trouble of
# making a tree from data every time during testing
def retrieveTree(i):
	listOfTrees = [{'no surfacing' : {0: 'no', 1: {'flippers': \
	                {0: 'no', 1: 'yes'}}}},
	               {'no surfacing' : {0: 'no', 1: {'flippers': \
	                {0: {'head' : {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
	              ]
	return listOfTrees[i]

# plots text between child and parent
def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString)

# the plotTree function
def plotTree(myTree, parentPt, nodeTxt):
	# get the width and height
	numLeafs = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)
	firstStr = myTree.keys()[0]
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, \
		                    plotTree.yOff)
	# plot child value
	plotMidText(cntrPt, parentPt, nodeTxt)
	plotNode(firstStr, cntrPt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
	# decrement Y offset
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			plotTree(secondDict[key], cntrPt, str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
				     cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD



