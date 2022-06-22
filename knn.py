import numpy as np
from scipy import stats
from abc import abstractmethod
from data import StandardScaler

class KNN:
    k = 0
    X_train = []
    y_train = []

    def __init__(self, k):
        self.k = k


    def fit(self, X_train, y_train):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.Y_train = scaler.fit_transform(y_train)


    @abstractmethod
    def predict(self, X_test):
            """ predict
            labels
            for X_test and return predicted labels """

    @staticmethod
    def dist(x1, x2):
        distance = np.linalg.norm(x1 - x2)
        return distance
        """returns
        Euclidean
        distance
        between
        x1 and x2"""


    def neighbours_indices(self, x):
        distances = np.empty()
        for observation in self.X_train:
            np.append(distances, self.dist(x , observation))
        indices = np.argpartition(distances, self.k)[:self.k]
        return indices
        """ for a given point x, find indices of k closest points in the training set """


class ClassificationKNN(KNN):
    def __init__(self, k):
        super().__init__(k)

        """ object
        instantiation, parent

        class instantiation """

    def predict(self, X_test):
        y_pred = np.empty()
        for point in X_test:
            np.append(y_pred, stats.mode(self.y_train[self.neighbours_indices(point)]))
        return y_pred
        """ predict
        labels
        for X_test and return predicted labels """


class RegressionKNN(KNN):

    def __init__(self, k):
        super().__init__(k)

        """ object
        instantiation, parent

        class instantiation """

    def predict(self, X_test):
        y_pred = np.empty()
        for point in X_test:
            np.append(y_pred, (self.y_train[self.neighbours_indices(point)]).mean())
        return y_pred
        """ predict
        labels
        for X_test and return predicted labels """





