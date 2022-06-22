import math

import numpy as np
import matplotlib.pyplot as plt


def binary_confusion_matrix(y_true, y_pred):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  TP = np.sum((y_true == 1) & (y_pred == 1))
  FN = np.sum((y_true == 1) & (y_pred == 0))
  FP = np.sum((y_true == 0) & (y_pred == 1))
  TN = np.sum((y_true == 0) & (y_pred == 0))

  return TN, FP, FN, TP


def f1_score(y_true, y_pred):
    """ returns f1_score of binary classification task with true labels y_true and predicted labels """
    TN, FP, FN, TP = binary_confusion_matrix(y_true, y_pred)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    return 2 * precision * recall/(recall + precision)


def rmse(y_true, y_pred):
    y_diff = y_true - y_pred
    return math.sqrt((y_diff*y_diff).sum/len(y_true))


def visualize_results(k_list, scores, metric_name, title, path):

