import numpy as np
import random


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
    def __init__(self, k):
        self.k = k

    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
        self.data = np.hstack((train_X, np.reshape(train_y, (-1, 1))))
        self.tree = self.find_tree(0, len(train_X) - 1, range(0, len(train_X[0]) - 1))

    def find_tree(self, start, stop, set_attributes):
        c = np.unique(self.train_y[start: stop])
        if len(c) == 1:
            return TerminalTreeNode(c[0])

        attribute, pivot = self.find_target_attribute(start, stop, set_attributes, c)
        pivot_value = self.train_X[pivot, attribute]
        set_attributes.remove(attribute)
        left = self.find_tree(start, pivot, set_attributes[:])
        right = self.find_tree(pivot + 1, stop, set_attributes[:])

        return DecisionTreeNode(attribute, pivot_value, left, right)

    def find_target_attribute(self, start, stop, set_attributes, c):
        best_f = 0
        best_gini = 0
        best_pivot = 0
        for i in set_attributes:
            gini, pivot = find_decision_boundary(self.data, start, stop, i, c)
            if gini > best_gini:
                best_gini = gini
                best_pivot = pivot
                best_f = i

        return best_f, best_pivot

    def evaluate(self, row):
        tree = self.tree
        while ~tree.isTerminal:
            tree = tree.evaluate(row)

        return tree.index
