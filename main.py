import sys

import data
from data import *
from cross_validation import *
from evaluation import *
from knn import *

def main(argv):
    k_list = [3, 5, 11, 25, 51, 75, 101]
    df = load_data(argv)
    folds = data.get_folds()

    # Part 1 - Classification
    X = add_noise(df['t1', 't2', 'wind_speed', 'hum'])
    y = adjust_labels(df['season'])
    means, standard_deviations = model_selection_cross_validation(ClassificationKNN, k_list, X, y, folds, f1_score)
    print("Part 1 - Classification")
    for i in len(range(k_list)):
        print(f"k = {k_list[i]}, mean score: {means[i]}, std of scores: {standard_deviations[i]}")

    print()
    # Part 2 - Regression
    X = add_noise(df['t1', 't2', 'wind_speed'])
    y = df['hum']
    means, standard_deviations = model_selection_cross_validation(RegressionKNN, k_list, X, y, folds, rmse)
    print("Part 2 - Regression")
    for i in len(range(k_list)):
        print(f"k = {k_list[i]}, mean score: {means[i]}, std of scores: {standard_deviations[i]}")


if __name__ == '__main__':
    main(sys.argv)