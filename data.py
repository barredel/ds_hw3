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
    df = pd.read_csv(path)
    return df


def adjust_labels(y):
    y_adjusted = list(map(lambda x: 0 if x <= 1 else 1, y))
    return y_adjusted


class StandardScaler:
    pass

    def __init__(self):



    def fit(self, X):
    “”” fit scaler by learning mean and standard deviation per feature “””


    def transform(self, X):
   ## “”” transform X by learned mean and standard deviation, and return it “””
    transformed_data = np.array([np.array(list(df[features[0]])), np.array(list(df[features[0]]))])
    print(transformed_data)
    sum0 = transformed_data[0].sum()
    min0 = transformed_data[0].min()
    sum1 = transformed_data[1].sum()
    min1 = transformed_data[1].min()
    (transformed_data[0] - min0)/sum0
    (transformed_data[1] - min1) / sum1
    print(transformed_data)


    def fit_transform(self, X):
    “”” fit scaler by learning mean and standard deviation per feature, and then transform X “””





