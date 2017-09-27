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


def swap(A, i, j):
    temp = A[i]
    A[i] = A[j]
    A[j] = temp


def calculate_gini(L, pivot_idx, f, c):
    l = np.zeros(len(c))
    m = np.zeros(len(c))
    for x in range(0, len(c)):
        l[x] = sum(1 for i in L[0:pivot_idx - 1, L.shape[1] - 1] if i == c[x])
        m[x] = sum(1 for i in L[pivot_idx:len(L), L.shape[1] - 1] if i == c[x])

    pm = sum(m) / (sum(m) + sum(l))
    pl = sum(l) / (sum(m) + sum(l))
    if sum(m) != 0:
        m = m / sum(m)

    if sum(l) != 0:
        l = l / sum(l)

    print("Calculating gini")
    gini = pl * sum(i ** 2 for i in l) + pm * sum(i ** 2 for i in m)
    print("Returning gini %.2f", gini)
    return gini, pivot_idx


def merge(left, right, f):
    if not len(left) or not len(right):
        return left or right

    result = np.empty([0, len(left[0])])
    i, j = 0, 0
    while (len(result) < len(left) + len(right)):
        if left[i][f] < right[j][f]:
            result = np.vstack((result, left[i]))
            i += 1
        else:
            result = np.vstack((result, right[j]))
            j += 1
        if i == len(left):
            result = np.vstack((result, right[j:]))
            break
        if j == len(right):
            result = np.vstack((result, left[i:]))
            break

    return result


def mergesort(list, f):
    if len(list) < 2:
        return list

    middle = len(list) // 2
    left = mergesort(list[:middle], f)
    right = mergesort(list[middle:], f)

    return merge(left, right, f)


def find_decision_boundary(L, start, stop, f, c):
    if stop - start < 2:
        return calculate_gini(L, start, f, c)

    pivot_idx = random.randint(start, stop)
    key = L[pivot_idx, f]
    e = u = start
    g = stop
    while u < g:
        if L[u, f] < key:
            swap(L, u, e)
            e = e + 1
            u = u + 1
        elif L[u, f] == key:
            u = u + 1
        else:
            g = g - 1
            swap(L, u, g)

    gini, pivot_idx = calculate_gini(L, e, f, c)
    if gini > 0.5:
        return gini, pivot_idx

    gini_left, idx_left = find_decision_boundary(L, start, e, f, c)
    if gini_left > 0.5:
        return gini_left, idx_left

    gini_right, idx_right = find_decision_boundary(L, g, stop, f, c)
    if gini_right > 0.5:
        return gini, idx_right

    max_gini = gini
    final_pivot = pivot_idx
    if gini_left > max_gini:
        max_gini = gini_left
        final_pivot = idx_left
    if gini_right > max_gini:
        max_gini = gini_right
        final_pivot = idx_right
    return max_gini, final_pivot


print("Reading Data")
nFeatures, nClasses, train_X, train_y = readChoirDat("mnist_train.choir_dat")
train_X = train_X[0:100]
train_y = train_y[0:100]
_, _, test_X, test_y = readChoirDat("mnist_test.choir_dat")
test_X = test_X[0:10]
test_y = test_y[0:10]

c = np.unique(train_y)
print("Start")

#mergesort(train_X, 256)
start = time.time()
#clf = ApproximateDecisionTreeClassifier(1, 3)
clf = DeterministicDecisionTreeClassifier(2, 3)
# clf = ApproximateDecisionTreeClassifier(3, 3)
# clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(train_X, train_y)

pred = clf.predict(test_X)
end = time.time()
print("Time Taken: %d", end - start)
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
