import sys
import data
from data import *
from cross_validation import *
from evaluation import *
from knn import *


def main(argv):
    k_list = [3, 5, 11, 25, 51, 75, 101]
    df = load_data(argv[1])
    folds = data.get_folds()

    # Part 1 - Classification
    X = add_noise(df[['t1', 't2', 'wind_speed', 'hum']].to_numpy())
    y = adjust_labels(df['season'].to_numpy())
    means, standard_deviations = model_selection_cross_validation(ClassificationKNN, k_list, X, y, folds, f1_score)
    print("Part1 - Classification")
    for i in range(len(k_list)):
        print(f"k={k_list[i]}" + ", mean score: " + "{:.4f}".format(round(means[i], 4)) + ", std of scores: "
              + "{:.4f}".format(round(standard_deviations[i], 4)))
    visualize_results(k_list, means, "f1 score", "Classification", "./classificationPlot.png")

    print()
    # Part 2 - Regression
    X = add_noise(df[['t1', 't2', 'wind_speed']].to_numpy())
    y = df['hum'].to_numpy()
    means, standard_deviations = model_selection_cross_validation(RegressionKNN, k_list, X, y, folds, rmse)
    print("Part2 - Regression")
    for i in range(len(k_list)):
        print(f"k={k_list[i]}" + ", mean score: " + "{:.4f}".format(round(means[i], 4)) + ", std of scores: "
              + "{:.4f}".format(round(standard_deviations[i], 4)))
    visualize_results(k_list, means, "RMSE", "Regression", "./regressionPlot.png")


if __name__ == '__main__':
    main(sys.argv)
