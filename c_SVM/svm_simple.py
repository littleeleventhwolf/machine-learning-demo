from __future__ import print_function

from sklearn import svm

X = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

print(clf)

# get support vector
print(clf.support_vectors_)

# get indices of support vectors
print(clf.support_)

# get number of support vectors for each class
print(clf.n_support_)