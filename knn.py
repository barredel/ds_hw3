import numpy as np
from scipy import stats
from abc import abstractmethod
from data import StandardScaler


class KNN:
    """
    represents KNN algorithm
    """
    k = 0
    X_train = []
    y_train = []
    scaler = 0

    def __init__(self, k):
        """
        class constructor
        :param k: number of neighbors
        """
        self.k = k
        self.X_train = []
        self.y_train = []
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        """
        fit scaler by learning mean and standard deviation per feature, and then transform X. saves y
        :param X_train: np array with train observations as rows and features as columns
        :param y_train: np array with labels for each train observation
        :return:
        """
        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train

    @abstractmethod
    def predict(self, X_test):
        """
        predict labels for X_test and return predicted labels or values
        :param X_test: np array with test observations as rows and features as columns
        :return: np array with prediction label or value for each train observation
        """

    @staticmethod
    def dist(x1, x2):
        """
        returns Euclidean distance between x1 and x2
        :param x1: a point
        :param x2: a point
        :return: the euclidian distance between x1 and x2
        """
        distance = np.linalg.norm(x1 - x2)
        return distance

    def neighbours_indices(self, x):
        """
        for a given point x, find indices of the k closest points in the training set
        :param x: a test single observation
        :return: indices of the k closest points in the training set
        """
        distances = []
        for observation in self.X_train:
            distances.append(self.dist(x, observation))
        distances = np.array(distances)
        indices = np.argpartition(distances, self.k)[:self.k]
        return indices


class ClassificationKNN(KNN):
    """
    represents classification KNN algorithm
    """

    def __init__(self, k):
        """
        class constructor
        :param k: k for KNN
        """
        super().__init__(k)

    def predict(self, X_test):
        """
        predict labels for X_test and return predicted labels
        :param X_test: np array with test observations as rows and features as columns
        :return: np array with prediction label for each train observation
        """
        X_test = self.scaler.transform(X_test)
        y_pred = []
        for point in X_test:
            m = stats.mode(self.y_train[self.neighbours_indices(point)])
            y_pred.append(m[0][0])
        y_pred = np.array(y_pred)
        return y_pred


class RegressionKNN(KNN):
    """
    represents regression KNN algorithm
    """

    def __init__(self, k):
        """
        class constructor
        :param k: k for KNN
        """
        super().__init__(k)

    def predict(self, X_test):
        """
        predict labels for X_test and return predicted values
        :param X_test: np array with test observations as rows and features as columns
        :return: np array with prediction value for each train observation
        """
        X_test = self.scaler.transform(X_test)
        y_pred = []
        for point in X_test:
            y_pred.append((self.y_train[self.neighbours_indices(point)]).mean())
        y_pred = np.array(y_pred)
        return y_pred
