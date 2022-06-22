import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.001^2)
    """
    noise = np.random.normal(loc=0, scale=0.001, size=data.shape)
    return data + noise


def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=34)


def load_data(path):
    """
    reads and returns the pandas DataFrame
    :param path: the path of the data file
    :return: the data organized in pandas DataFrame
    """
    df = pd.read_csv(path)
    return df


def adjust_labels(y):
    """
    changes 0 and 1 (spring and summer) values to 0, 2 and 3 (fall and winter) values to 1
    :param y: an array of values from 'season' column in the DataFrame
    :return: array of values the same size as y, with 0 instead spring and summer, and 1 instead fall and winter
    """
    y_adjusted = np.array(list(map(lambda x: 0 if x <= 1 else 1, y)))
    return y_adjusted


class StandardScaler:
    """
    represents a scaler by specific standard_deviation and mean
    """
    standard_deviations = []
    means = []


    def __init__(self):
        """
        class constructor
        """
        standard_deviations = []
        means = []

    def fit(self, X):
        """
        fit scaler by learning mean and standard deviation per feature
        :param X: np array with observations as rows and features as columns
        """
        self.standard_deviations = np.array(np.std(X, axis=0, ddof=1))
        self.means = np.array((np.mean(X, axis=0)))

    def transform(self, X):
        """
        transform X by learned mean and standard deviation, and return it
        :param X: np array with observations as rows and features as columns
        :return: X transformed by mean and standard deviation
        """
        return (X-self.means)/self.standard_deviations

    def fit_transform(self, X):
        """
        fit scaler by learning mean and standard deviation per feature, and then transform X
        :param X: np array with observations as rows and features as columns
        :return: X transformed by mean and standard deviation
        """
        self.fit(X)
        return self.transform(X)
