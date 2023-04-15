import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from PIL import Image

# import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from functools import partial
import sys


def error_custom(y_val, y_pred, err_type, threshold=5):
    """error which only takes into account buildings > 5m"""
    y_val_copy = y_val[np.where(y_val >= threshold)]
    y_pred_copy = y_pred[np.where(y_val >= threshold)]

    print(np.size(y_val), np.size(y_val_copy))

    if err_type == "mae":
        return -mean_absolute_error(y_val_copy, y_pred_copy)
    elif err_type == "mse":
        return -mean_squared_error(y_val_copy, y_pred_copy)
    elif err_type == "mape":
        return -100 * mean_absolute_percentage_error(y_val_copy, y_pred_copy)


def custom_acc(Y_test, Y_pred, penalty_wrong=True):
    bin_upper_bounds = np.arange(11)
    Y_test_bins = np.zeros((len(Y_test)),dtype=int)
    Y_pred_bins = np.zeros((len(Y_pred)),dtype=int)
    totscore = 0
    for i in range(len(Y_test)):
        for j in range(11):
            if Y_test[i] >= bin_upper_bounds[j]:
                Y_test_bins[i] = j
            if Y_pred[i] >= bin_upper_bounds[j]:
                Y_pred_bins[i] = j
        scores_diff = np.abs(Y_test_bins[i] - Y_pred_bins[i])
        if scores_diff == 0:
            totscore += 1.
        elif scores_diff == 1:
            totscore += 2./3.
        elif scores_diff == 2:
            totscore += 1./3
        if penalty_wrong:
            if scores_diff == 4:
                totscore -= 1/.3
            elif scores_diff == 5:
                totscore -=2./3
            else:
                totscore -=1
    avscore = totscore / len(Y_test)
    return avscore
    
# first we load in the data
big_data = np.load("/work/unicef/pixels_features_shuffled.npy")
#big_data = np.load("/work/unicef/features_balanced_v2.npy")
# print(np.shape(big_data))

# take a random subset if desired
subset_frac = float(sys.argv[1])
nrows = int(subset_frac * np.shape(big_data)[0])
data = big_data[:nrows]

X = data[:, :-1]
Y = data[:, -1]

# scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y.ravel(), test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

print("Accuracy = ", custom_acc(Y_test, Y_pred_test, penalty_wrong=False))
print("MSE (all data) = ", mean_squared_error(Y_test, Y_pred_test))
print("MAE (all data) = ", mean_absolute_error(Y_test, Y_pred_test))
print("R2 score (all data) = ", r2_score(Y_test, Y_pred_test))
print("MSE (>10 m) = ", -error_custom(Y_test, Y_pred_test, err_type="mse", threshold=10))
print("MAE (>10 m) = ", -error_custom(Y_test, Y_pred_test, err_type="mae", threshold=10))

np.savez("Randomforest_preds.npy", Y_test, Y_pred_test)

# cross validated grid search
# define the parameter search space
# we will go through various combinations
# param_grid = {"max_depth": [5,20],
#               "min_samples_split": [4,8],
#               "min_samples_leaf": [4,8],
#               "n_estimators": [50,500],
#               "max_leaf_nodes": [10,100]}


#param_grid = {"max_depth": [5, 30, None]}
param_grid = {"max_depth": [None], "min_samples_split": [2], "n_estimators": [50,200]}

#param_grid_list = [{"max_depth": [3, 5, 10]}]

scoring_dict = {
    "mae_5": make_scorer(
        lambda y_val, y_pred: error_custom(y_val, y_pred, err_type="mae"),
        greater_is_better=False,
    ),
    "mse_5": make_scorer(
        lambda y_val, y_pred: error_custom(y_val, y_pred, err_type="mse"),
        greater_is_better=False,
    ),
    "mape_5": make_scorer(
        lambda y_val, y_pred: error_custom(y_val, y_pred, err_type="mape"),
        greater_is_better=False,
    ),
    "mae": make_scorer(
        lambda y_val, y_pred: error_custom(y_val, y_pred, err_type="mae", threshold=0),
        greater_is_better=False,
    ),
    "mse": make_scorer(
        lambda y_val, y_pred: error_custom(y_val, y_pred, err_type="mse", threshold=0),
        greater_is_better=False,
    ),        
}

scoring_dict = {
    "acc_no_pen": make_scorer(
        lambda y_val, y_pred: custom_acc(y_val, y_pred, penalty_wrong=False),
        greater_is_better=True,
    ),
    "acc_pen": make_scorer(
        lambda y_val, y_pred: custom_acc(y_val, y_pred, penalty_wrong=True),
        greater_is_better=True,
    ),
    "mape_1": make_scorer(
        lambda y_val, y_pred: error_custom(y_val, y_pred, err_type="mape", threshold=1),
        greater_is_better=False,
    ),
    "mae": make_scorer(
        lambda y_val, y_pred: error_custom(y_val, y_pred, err_type="mae", threshold=0),
        greater_is_better=False,
    ),
    "mse": make_scorer(
        lambda y_val, y_pred: error_custom(y_val, y_pred, err_type="mse", threshold=0),
        greater_is_better=False,
    ),        
}



grid_search = GridSearchCV(
    model,
    param_grid,
    scoring=scoring_dict,
    verbose=4,
    return_train_score=True,
    refit="acc_pen",
)
# grid_search.fit(X_train, Y_train)
# df = pd.DataFrame.from_dict(grid_search.cv_results_)
# df.to_csv("rf_scores_" + str(subset_frac) + ".csv")
