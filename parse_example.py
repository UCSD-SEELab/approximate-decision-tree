import sys
import struct
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
import time
import random
import math
import numpy as np
from decision_tree import ApproximateDecisionTreeClassifier
from decision_tree import DeterministicDecisionTreeClassifier
from operator import add
from scipy import stats
from collections import Counter


def run_adaboost(train_X, train_y, test_X, test_y):
    c = np.unique(train_y)
    n_classes = len(c)
    num = 10
    clfs = []
    for loop_idx in range(0, num):
        clfs.append(ApproximateDecisionTreeClassifier(3, 3))

    weights = [1] * len(train_X)
    t = sum(weights)
    for i in range(0, len(weights)):
        weights[i] = weights[i] / t

    errors = [0] * len(clfs)
    alphas = [0] * len(clfs)
    for loop_idx in range(0, num):
        clf = clfs[loop_idx]
        clf.fit(train_X, train_y, weights)
        pred = clf.predict(train_X)
        accuracy = metrics.accuracy_score(train_y, pred)
        print("Accuracy for adaboost loop %d: %.2f" % (loop_idx, accuracy))
        pred_prob_i = clf.predict_proba(train_X)
        error = 0
        for i in range(0, len(pred)):
            if pred[i] == train_y[i]:
                error = error + (weights[i] * (1 - pred_prob_i[i, int(train_y[i])]))

        error = error / sum(weights)
        alpha = math.log((1 - error) / error) + math.log(1 + n_classes)
        alphas[loop_idx] = alpha
        errors[loop_idx] = error
        for i in range(0, len(pred)):
            if weights[i] > 0:
                weights[i] = weights[i] * math.exp((1 - pred_prob_i[i, int(train_y[i])]) * alpha)

        total = sum(weights)
        if total != 0:
            for i in range(0, len(weights)):
                weights[i] = weights[i] / total
                # weights = weights * math.exp(alpha*identi)

        # testing
        pred_prob = np.zeros((len(train_X), n_classes))
        for loop_idx_temp in range(0, loop_idx):
            clf = clfs[loop_idx_temp]
            pred_prob = np.add(pred_prob, alphas[loop_idx_temp] * np.array(clf.predict_proba(train_X)))

        pred_final = [0] * len(train_X)
        for loop_idx_k in range(0, len(train_X)):
            list_i = pred_prob[loop_idx_k]
            pred_final[loop_idx_k] = np.argmax(list_i)

        accuracy = metrics.accuracy_score(train_y, pred_final)
        print("Accuracy for adaboost loop end %d: %.2f" % (loop_idx, accuracy))

    pred_prob = np.zeros((len(test_X), n_classes))
    for loop_idx in range(0, num):
        clf = clfs[loop_idx]
        pred_prob = np.add(pred_prob, alphas[loop_idx] * np.array(clf.predict_proba(test_X)))
        # pred[:, loop_idx] = clf.predict(test_X)
        # pred_i = [i * alphas[loop_idx] for i in pred]
        # map(add, pred, pred_i)

    pred_final = [0] * len(test_X)
    for loop_idx in range(0, len(test_X)):
        list_i = pred_prob[loop_idx]
        pred_final[loop_idx] = np.argmax(list_i)

    pred = pred_final

    accuracy = metrics.accuracy_score(test_y, pred)
    print("Accuracy for adaboost: %.2f" % accuracy)


def run_decision_tree(clf, train_X, train_y, test_X, test_y):
    start = time.time()
    clf.fit(train_X, train_y, [1] * len(train_X))
    end = time.time()
    print("Time Taken for training: %d", end - start)

    pred_prob = clf.predict_proba(test_X)
    pred = [0] * len(test_X)
    for loop_idx in range(0, len(test_X)):
        list_i = pred_prob[loop_idx]
        pred[loop_idx] = np.argmax(list_i)
    accuracy = metrics.accuracy_score(test_y, pred)
    print("Accuracy: %.2f" % accuracy)


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
nFeatures, nClasses, train_X, train_y = readChoirDat("dataset/face/face_train.choir_dat")
#nFeatures, nClasses, train_X, train_y = readChoirDat("dataset/MNIST/mnist_hog44_train.choir_dat")
#nFeatures, nClasses, train_X, train_y = readChoirDat("dataset/MNIST/mnist_train.choir_dat")
#nFeatures, nClasses, train_X, train_y = readChoirDat("dataset/iris/iris_train.choir_dat")
#nFeatures, nClasses, train_X, train_y = readChoirDat("dataset/PAMPA2/PAMPA2_processed_train.choir_dat")
train_X = train_X[0:100]
train_y = train_y[0:100]
_, _, test_X, test_y = readChoirDat("dataset/face/face_test.choir_dat")
#_, _, test_X, test_y = readChoirDat("dataset/MNIST/mnist_test.choir_dat")
#_, _, test_X, test_y = readChoirDat("dataset/iris/iris_test.choir_dat")
#_, _, test_X, test_y = readChoirDat("dataset/PAMPA2/PAMPA2_processed_test.choir_dat")
# test_X = test_X[0:100]
# test_y = test_y[0:100]



print("Start")
clf = ApproximateDecisionTreeClassifier(3, 3)
run_adaboost(train_X, train_y, test_X, test_y)

# clf = DeterministicDecisionTreeClassifier(2, 3)
# clf = ApproximateDecisionTreeClassifier(3, 3)
# clf = tree.DecisionTreeClassifier(max_depth=3)
