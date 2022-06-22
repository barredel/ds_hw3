import math

import numpy as np
import matplotlib.pyplot as plt


def binary_confusion_matrix(y_true, y_pred):
    """
    creates a confusion matrix and returns each value of the matrix
    :param y_true: true labels of data
    :param y_pred: predicted labels of data
    :return: returns each value of the confusion matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    return TN, FP, FN, TP


def f1_score(y_true, y_pred):
    """
    calculates f1 score of the prediction, and returns it
    :param y_true: true labels of data
    :param y_pred: predicted labels of data
    :return: f1 score of the prediction
    """
    TN, FP, FN, TP = binary_confusion_matrix(y_true, y_pred)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    return 2 * precision * recall/(recall + precision)


def rmse(y_true, y_pred):
    """
    calculates RMSE of the prediction, and returns it
    :param y_true: true labels of data
    :param y_pred: predicted labels of data
    :return: RMSE of the prediction
    """
    y_diff = y_true - y_pred
    return math.sqrt((y_diff*y_diff).sum()/len(y_true))


def visualize_results(k_list, scores, metric_name, title, path):
    """
    creates a graph of k's and their cross validation mean results from KNN
    :param k_list: list of k's for KNN
    :param scores: list of cross validation mean results of each k
    :param metric_name: string with the name of the metric that was used in calculation
    :param title: the title of the graph
    :param path: the path to save the graph in
    """
    plt.plot(k_list, scores)
    plt.xlabel("k")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.savefig(path)
    plt.show()
