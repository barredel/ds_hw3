import numpy as np
from scipy import stats
from abc import abstractmethod
from data import StandardScaler


class KNN:
    k = 0
    X_train = []
    y_train = []
    scaler = 0

    def __init__(self, k):
        self.k = k
        self.X_train = []
        self.y_train = []
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train

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
        distances = []
        for observation in self.X_train:
            distances.append(self.dist(x, observation))
        distances = np.array(distances)
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
        X_test = self.scaler.transform(X_test)
        y_pred = []
        for point in X_test:
            m = stats.mode(self.y_train[self.neighbours_indices(point)])
            y_pred.append(m[0][0])
        y_pred = np.array(y_pred)
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
        X_test = self.scaler.transform(X_test)
        y_pred = []
        for point in X_test:
            y_pred.append((self.y_train[self.neighbours_indices(point)]).mean())
        y_pred = np.array(y_pred)
        return y_pred
        """ predict
        labels
        for X_test and return predicted labels """
