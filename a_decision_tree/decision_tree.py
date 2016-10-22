from __future__ import print_function

import numpy as np
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

# read in the csv file and put features in a list of dict and list of class label
allElectronicsData = open(r'data.csv', 'rb')
reader = csv.reader(allElectronicsData)
headers = reader.next()

# print(headers)

featureList = []
labelList = []

for row in reader:
	labelList.append(row[-1])
	rowDict = {}
	for i in range(1, len(row) - 1):
		rowDict[headers[i]] = row[i]
	featureList.append(rowDict)

print(featureList)

# Vectorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print("dummyX: " + str(dummyX))
print(vec.get_feature_names())

print("LabelList: " + str(labelList))

# Vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))

# Using decision tree for classification
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))

# Visualize model
with open("allElectronicsInformationGain.dot", 'w') as f:
	f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX

newRowX[0] = 1
newRowX[2] = 0
print('newRowX: ' + str(newRowX))

# To solve the DeprecationWarning : method 1
# newRowX = np.array(newRowX)
# print('newRowX: ' + str(newRowX))
# newRowX = newRowX.reshape((1, -1))
# print('newRowX: ' + str(newRowX))
# predictedY = clf.predict(newRowX)

# method 2
predictedY = clf.predict([newRowX])


print("predictedY: " + str(predictedY))