import numpy as np


def cross_validation_score(model, X, y, folds, metric):
    """ run cross validation on X and y with specific model by given folds. Evaluate by given metric. """
    scores = np.empty()
    for train_indices, validation_indices in folds.split(X):
        model.fit(X[train_indices], y[train_indices])
        y_pred = model.predict(X[validation_indices])
        np.append(scores, metric(y[validation_indices], y_pred))
    return scores

def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    means = np.empty()
    standard_deviations = np.empty()
    for k in k_list:
        knn = model(k)
        np.append(means, cross_validation_score(knn, X, y, folds, metric).mean())
        np.append(standard_deviations, cross_validation_score(knn, X, y, folds, metric).std())
    return means, standard_deviations

