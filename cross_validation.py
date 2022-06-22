import numpy as np


def cross_validation_score(model, X, y, folds, metric):
    """ run cross validation on X and y with specific model by given folds. Evaluate by given metric. """