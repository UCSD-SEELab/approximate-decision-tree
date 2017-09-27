import sys
import struct
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
import time
import random
import numpy as np
from decision_tree import ApproximateDecisionTreeClassifier
from decision_tree import DeterministicDecisionTreeClassifier


def readChoirDat(filename):
    """ Parse a choir_dat file """
    with open(filename, 'rb') as f:
        nFeatures = struct.unpack('i', f.read(4))[0]
        nClasses = struct.unpack('i', f.read(4))[0]

        X = []
        y = []

        while True:
            newDP = []
            for i in range(nFeatures):
                v_in_bytes = f.read(4)
                if v_in_bytes is None or len(v_in_bytes) == 0:
                    return nFeatures, nClasses, X, y

                v = struct.unpack('f', v_in_bytes)[0]
                newDP.append(v)

            l = struct.unpack('i', f.read(4))[0]
            X.append(newDP)
            y.append(l)

    return nFeatures, nClasses, X, y


print("Reading Data")
nFeatures, nClasses, train_X, train_y = readChoirDat("dataset/iris/iris_train.choir_dat")
#train_X = train_X[0:1000]
#rain_y = train_y[0:1000]
_, _, test_X, test_y = readChoirDat("dataset/iris/iris_test.choir_dat")
#test_X = test_X[0:100]
#test_y = test_y[0:100]

c = np.unique(train_y)

print("Start")
start = time.time()
clf = ApproximateDecisionTreeClassifier(1, 3)
clf.fit(train_X, train_y, [1]*len(train_X))
end = time.time()
print("Time Taken for training approx: %d", end - start)

pred = clf.predict(test_X)
accuracy = metrics.accuracy_score(test_y, pred)
print("Accuracy approx: %.2f" % accuracy)

clf = DeterministicDecisionTreeClassifier(2, 3)
# clf = ApproximateDecisionTreeClassifier(3, 3)
# clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(train_X, train_y, [1]*len(train_X))
end = time.time()
print("Time Taken for training: %d", end - start)

pred = clf.predict(test_X)
accuracy = metrics.accuracy_score(test_y, pred)
print("Accuracy: %.2f" % accuracy)

# gini, pivot = find_decision_boundary(np.hstack((train_X, np.reshape(train_y, (-1, 1)))), 0, len(train_X)-1, 125, c)

# print("gini: %.2f" % gini)
# start = time.time()
# clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1),
#                                  algorithm="SAMME",
#                                  n_estimators=200)
# clf = clf.fit(train_X, train_y)
# y_pred = clf.predict(test_X)

# accuracy = metrics.accuracy_score(test_y, y_pred)
# end = time.time()
# print("Time Taken: %d", end - start)
# print("Accuracy: %.2f" % accuracy)
