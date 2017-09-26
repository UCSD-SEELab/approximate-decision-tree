import numpy as np
import random


def swap(A, i, j):
    A[[i, j], :] = A[[j, i], :]


def calculate_gini(L, pivot_idx, f, c):
    l = np.zeros(len(c))
    m = np.zeros(len(c))
    for x in range(0, len(c)):
        l[x] = sum(1 for i in L[0:pivot_idx - 1, L.shape[1] - 1] if i == c[x])
        m[x] = sum(1 for i in L[pivot_idx:len(L), L.shape[1] - 1] if i == c[x])

    pm = sum(m)
    pl = sum(l)

    if (sum(m) + sum(l)) != 0:
        pm = pm / (sum(m) + sum(l))
        pl = pl / (sum(m) + sum(l))

    if sum(m) != 0:
        m = m / sum(m)

    if sum(l) != 0:
        l = l / sum(l)

    gini = pl * sum(i ** 2 for i in l) + pm * sum(i ** 2 for i in m)
    return gini, pivot_idx


def find_median(A):
    return median_of_medians(A, len(A) // 2)


def median_of_medians(A, i):
    # divide A into sublists of len 5
    sublists = [A[j:j + 5] for j in range(0, len(A), 5)]
    medians = [sorted(sublist)[len(sublist) // 2] for sublist in sublists]
    if len(medians) <= 5:
        pivot = sorted(medians)[len(medians) // 2]
    else:
        # the pivot is the median of the medians
        pivot = median_of_medians(medians, len(medians) // 2)

    # partitioning step
    low = [j for j in A if j < pivot]
    high = [j for j in A if j > pivot]

    k = len(low)
    if i < k:
        return median_of_medians(low, i)
    elif i > k:
        return median_of_medians(high, i - k - 1)
    else:  # pivot = k
        return pivot


def find_decision_boundary(L, start, stop, f, c, depth):
    if stop - start < 2:
        return calculate_gini(L, start, f, c)

    # median = find_median(np.unique(L[:, f]))
    # pivot_idx = np.where(L[:, f] == median)[0][0]
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
            swap(L, u, g)
            g = g - 1

    gini, pivot_idx = calculate_gini(L, e, f, c)
    if gini > 0.5 or depth < 1:
        return gini, pivot_idx

    gini_left, idx_left = find_decision_boundary(L, start, e, f, c, depth - 1)
    if gini_left > 0.5 or depth < 1:
        return gini_left, idx_left

    gini_right, idx_right = find_decision_boundary(L, g, stop, f, c, depth - 1)
    if gini_right > 0.5 or depth < 1:
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


class DecisionTreeNode:
    def __init__(self, f, pivot_value, left, right):
        self.f = f
        self.pivot_value = pivot_value
        self.left = left
        self.right = right
        self.isTerminal = False

    def evaluate(self, row):
        if row[self.f] <= self.pivot_value:
            return self.left

        return self.right


class TerminalTreeNode:
    def __init__(self, index):
        self.index = index
        self.isTerminal = True


class ApproximateDecisionTreeClassifier:
    def __init__(self, k, depth):
        self.k = k
        self.depth = depth

    def fit(self, train_X, train_y):
        self.data = np.hstack((train_X, np.reshape(train_y, (-1, 1))))
        self.tree = self.find_tree(self.data, list(range(0, len(train_X[0]) - 1)), self.depth)

    def find_tree(self, data, set_attributes, depth):
        if len(data) < 1:
            print("zero elements")
            return TerminalTreeNode(0)

        col_y = len(self.data[0]) - 1

        if len(data) == 1:
            print("Start == stop")
            return TerminalTreeNode(data[0, col_y])

        c, counts = np.unique(data[:, col_y], return_counts=True)

        if len(c) == 1 or len(set_attributes) == 0:
            print("length of c %d" % len(c))
            print("length of attributes %d" % len(set_attributes))
            return TerminalTreeNode(c[0])

        if depth < 1:
            ind = np.argmax(counts)
            return TerminalTreeNode(c[ind])

        attribute, pivot = self.find_target_attribute(data, set_attributes, c)
        print("Choosing attribute: %d" % attribute)
        print("Choosing pivot: %d" % pivot)

        # fix the data
        attribute, pivot = self.find_target_attribute(data, [attribute], c)

        pivot_value = data[pivot][attribute]
        print("Choosing pivot value: %.2f" % pivot_value)
        set_attributes.remove(attribute)
        print("Going left")

        left_partition = data[data[:, attribute] <= pivot_value]
        right_partition = data[data[:, attribute] > pivot_value]
        left = self.find_tree(left_partition, set_attributes[:], depth - 1)
        print("Going right")
        right = self.find_tree(right_partition, set_attributes[:], depth - 1)

        return DecisionTreeNode(attribute, pivot_value, left, right)

    def find_target_attribute(self, data, set_attributes, c):
        best_f = 0
        best_gini = 0
        best_pivot = 0
        for i in set_attributes:
            gini, pivot = find_decision_boundary(data, 0, len(data)-1, i, c, self.k)
            if gini >= best_gini:
                best_gini = gini
                best_pivot = pivot
                best_f = i

        return best_f, best_pivot

    def predict(self, test_X):
        pred = [0] * len(test_X)
        for i in range(0, len(test_X)):
            pred[i] = self.evaluate(test_X[i])

        return pred

    def evaluate(self, row):
        tree = self.tree

        while not tree.isTerminal:
            tree = tree.evaluate(row)

        return tree.index
