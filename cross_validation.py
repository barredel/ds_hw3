import numpy as np


def cross_validation_score(model, X, y, folds, metric):
    """ run cross validation on X and y with specific model by given folds. Evaluate by given metric. """
    scores = [] #TODO FIX
    for train_indices, validation_indices in folds.split(X):
        model.fit(X[train_indices], y[train_indices])
        y_pred = model.predict(X[validation_indices])
        scores.append(metric(y[validation_indices], y_pred))
    return np.array(scores)

def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    means = []
    standard_deviations = []
    for k in k_list:
        knn = model(k)
        means.append(cross_validation_score(knn, X, y, folds, metric).mean())
        standard_deviations.append(cross_validation_score(knn, X, y, folds, metric).std(ddof=1))
    means = np.array(means)
    standard_deviations = np.array(standard_deviations)
    return means, standard_deviations

