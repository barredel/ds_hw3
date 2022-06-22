import numpy as np


def cross_validation_score(model, X, y, folds, metric):
    """
    run cross validation on X and y with specific model by given folds. Evaluate by given metric.
    :param model: an instance of a KNN model (classification or regression)
    :param X: np array with observations as rows and features as columns
    :param y: np array with labels for each observation
    :param folds: KFold object with the division of the data to separate folds
    :param metric: a metric to calculate the performance of the KNN (RMSE or f1 score)
    :return: an array of the scores by the metric
    """
    scores = []
    for train_indices, validation_indices in folds.split(X):
        model.fit(X[train_indices], y[train_indices])
        y_pred = model.predict(X[validation_indices])
        scores.append(metric(y[validation_indices], y_pred))
    return np.array(scores)


def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    """
    run cross validation on X and y for every model induced by values from k_list by given folds.
    Evaluate each model by given metric
    :param model: a class of KNN model (classification or regression)
    :param k_list: list of k's for KNN
    :param X: np array with observations as rows and features as columns
    :param y: np array with labels for each observation
    :param folds: KFold object with the division of the data to separate folds
    :param metric: a metric to calculate the performance of the KNN (RMSE or f1 score)
    :return:
    """
    means = []
    standard_deviations = []
    for k in k_list:
        knn = model(k)
        means.append(cross_validation_score(knn, X, y, folds, metric).mean())
        standard_deviations.append(cross_validation_score(knn, X, y, folds, metric).std(ddof=1))
    means = np.array(means)
    standard_deviations = np.array(standard_deviations)
    return means, standard_deviations
