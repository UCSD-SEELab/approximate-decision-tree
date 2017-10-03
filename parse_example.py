import sys
import struct
from sklearn import tree as t_ada
from sklearn import ensemble
from sklearn import metrics
import time
import random
import math
import numpy as np
from decision_tree import ApproximateDecisionTreeClassifier
from decision_tree import DeterministicDecisionTreeClassifier
import scipy
import pickle
from operator import add
from scipy import stats
from collections import Counter
from os import listdir
from os.path import isfile, join, splitext
from skimage.feature import hog
import xml.etree.ElementTree as ET


def run_adaboost(train_X, train_y, test_X, test_y):
    c = np.unique(train_y)
    n_classes = len(c)
    num = 10
    clfs = []
    for loop_idx in range(0, num):
        clfs.append(ApproximateDecisionTreeClassifier(3, 2))

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
        for loop_idx_temp in range(0, loop_idx + 1):
            clf = clfs[loop_idx_temp]
            pred_prob = np.add(pred_prob, alphas[loop_idx_temp] * np.array(clf.predict_proba(train_X)))

        pred_final = [0] * len(train_X)
        for loop_idx_k in range(0, len(train_X)):
            list_i = pred_prob[loop_idx_k]
            pred_final[loop_idx_k] = np.argmax(list_i)

        accuracy = metrics.accuracy_score(train_y, pred_final)
        print("Accuracy for adaboost loop end %d: %.2f" % (loop_idx, accuracy))

        pred_prob_test_1 = np.zeros((len(test_X), n_classes))
        for loop_idx_temp_1 in range(0, loop_idx + 1):
            clf = clfs[loop_idx_temp_1]
            pred_prob_test_1 = np.add(pred_prob_test_1, alphas[loop_idx_temp_1] * np.array(clf.predict_proba(test_X)))

        pred_final_test_1 = [0] * len(test_X)
        for loop_idx_1 in range(0, len(test_X)):
            list_i = pred_prob_test_1[loop_idx_1]
            pred_final_test_1[loop_idx_1] = np.argmax(list_i)

        accuracy = metrics.accuracy_score(test_y, pred_final_test_1)
        print("Accuracy for adaboost test loop end %d: %.2f" % (loop_idx, accuracy))

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


def get_annotated_names(annotationId):
    annotationId = "bbox/" + annotationId
    annotations = [splitext(f)[0] for f in listdir(annotationId) if isfile(join(annotationId, f))]
    return annotations


list_ids = ["n02100236/", "n02124075/"]
# "n02100236/" Dogs
# "n00017222/" plants
# "n00442437/"
# "n11596108/"
# "n02124075/" cats
N = 100
n_classes = len(list_ids)
test_N = 10

attribute_size = 1568
train_X = np.zeros((N * n_classes, attribute_size))
train_y = np.zeros((N * n_classes))
test_X = np.zeros((test_N * n_classes, attribute_size))
test_y = np.zeros((test_N * n_classes))
image_size = 64
total = 0
test_total = 0
cell_size = 8
for num in range(0, n_classes):
    folderId = list_ids[num]
    images = get_annotated_names(folderId)
    images = [image for image in images if isfile(folderId + image + ".JPEG")]
    for i in range(1, N + 1):
        fileName = images[i - 1]
        arr = scipy.misc.imread(folderId + fileName + ".JPEG", flatten=True)

        tree = ET.parse("bbox/" + folderId + fileName + ".xml")
        root = tree.getroot()
        size = root[3]
        width = int(size[0].text)
        height = int(size[1].text)
        bndbox = root[5][4]
        xmin = int(bndbox[0].text)
        ymin = int(bndbox[1].text)
        xmax = int(bndbox[2].text)
        ymax = int(bndbox[3].text)

        arr = scipy.misc.imresize(arr, (height, width))
        arr = arr[ymin:ymax, xmin:xmax]

        arr = scipy.misc.imresize(arr, (image_size, image_size))
        arr = hog(arr, orientations=8, pixels_per_cell=(cell_size, cell_size), cells_per_block=(2, 2),
                  block_norm='L2-Hys')
        # pixels_per_cell = (cell_size, cell_size),cells_per_block = (2, 2), block_norm='L2-Hys'
        train_X[total + i - 1] = np.array(arr).reshape((1, -1))
        train_y[total + i - 1] = num
        # img = scipy.misc.toimage(arr)
        # img.show()

    total = total + N

    for i in range(1, test_N + 1):
        fileName = images[N + i - 1]
        arr = scipy.misc.imread(folderId + fileName + ".JPEG", flatten=True)
        arr = scipy.misc.imresize(arr, (image_size, image_size))
        arr = hog(arr, orientations=8, pixels_per_cell=(cell_size, cell_size), cells_per_block=(2, 2),
                  block_norm='L2-Hys')
        test_X[test_total + i - 1] = np.array(arr).reshape((1, -1))
        test_y[test_total + i - 1] = num

    test_total = test_total + test_N

print("Reading Data")
# nFeatures, nClasses, train_X, train_y = readChoirDat("dataset/face/face_train.choir_dat")
# nFeatures, nClasses, train_X, train_y = readChoirDat("dataset/MNIST/mnist_hog44_train.choir_dat")
# nFeatures, nClasses, train_X, train_y = readChoirDat("dataset/MNIST/mnist_train.choir_dat")
# nFeatures, nClasses, train_X, train_y = readChoirDat("dataset/iris/iris_train.choir_dat")
# nFeatures, nClasses, train_X, train_y = readChoirDat("dataset/PAMPA2/PAMPA2_processed_train.choir_dat")
# train_X = train_X[0:100]
# train_y = train_y[0:100]
# _, _, test_X, test_y = readChoirDat("dataset/face/face_test.choir_dat")
# _, _, test_X, test_y = readChoirDat("dataset/MNIST/mnist_test.choir_dat")
# _, _, test_X, test_y = readChoirDat("dataset/iris/iris_test.choir_dat")
# _, _, test_X, test_y = readChoirDat("dataset/PAMPA2/PAMPA2_processed_test.choir_dat")
# test_X = test_X[0:100]
# test_y = test_y[0:100]


print("Start")
# clf = ApproximateDecisionTreeClassifier(3, 3)

at = 0
if at:
    run_adaboost(train_X, train_y, test_X, test_y)
else:
    clf = ensemble.AdaBoostClassifier(t_ada.DecisionTreeClassifier(max_depth=1),
                                      algorithm="SAMME",
                                      n_estimators=1000)
    clf = clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)

    accuracy = metrics.accuracy_score(test_y, y_pred)
    # end = time.time()
    # print("Time Taken: %d", end - start)
    print("Accuracy: %.2f" % accuracy)
# clf = DeterministicDecisionTreeClassifier(2, 3)
# clf = ApproximateDecisionTreeClassifier(3, 3)
# clf = tree.DecisionTreeClassifier(max_depth=3)
