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

# from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
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


# first we load in the data
big_data = np.load("/work/unicef/pixels_features_shuffled.npy")
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
    X, Y.ravel(), test_size=0.2, random_state=1
)

model = DecisionTreeRegressor()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

#for i in range(np.shape(X)[1]):
#     print(model.feature_importances_[i])


# print("Mean Squared Error (test): {:.2f}".format(mean_squared_error(Y_test, Y_pred_test)))
# print("Mean Squared Error (train): {:.2f}".format(mean_squared_error(Y_train, Y_pred_train)))
# print("Mean Absolute Error (test): {:.2f}".format(mean_absolute_error(Y_test, Y_pred_test)))
# print("Mean Absolute Error (train): {:.2f}".format(mean_absolute_error(Y_train, Y_pred_train)))
# loss_func = partial(error_custom, err_type="mse")
# print("Custom Error (MSE) (test): {:.2f}".format(loss_func(Y_test, Y_pred_test)))
# print("Custom Error (MSE) (train): {:.2f}".format(loss_func(Y_train, Y_pred_train)))
# print("Custom Error (MAE) (test): {:.2f}".format(error_custom(Y_test, Y_pred_test, "mae")))
# print("Custom Error (MAE) (train): {:.2f}".format(error_custom(Y_train, Y_pred_train, "mae")))
# print("Custom Error (MAPE) (test): {:.2f}".format(error_custom(Y_test, Y_pred_test, "mape")))
# print("Custom Error (MAPE) (train): {:.2f}".format(error_custom(Y_train, Y_pred_train, "mape")))


# cross validated grid search
# define the parameter search space
# we will go through various combinations
param_grid_list = [{"max_depth": [3, 5, 10, 20, 30, 50]},
                   {"min_samples_split": [1,2,4,8]},
                   {"min_samples_leaf": [1,2,4,8]}]

#param_grid_list = [{"max_depth": [3, 5, 10]}]

names = ["max_depth", "min_samples_split", "min_samples_leaf"]

for i, param_grid in enumerate(param_grid_list):
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

    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring=scoring_dict,
        verbose=4,
        return_train_score=True,
        refit=False,
    )
    grid_search.fit(X_train, Y_train)
    df = pd.DataFrame.from_dict(grid_search.cv_results_)
    df_new = df[
        [
            "params",
            "mean_train_mse",
            "mean_train_mae",
            "mean_train_mse_5",
            "mean_train_mae_5",
            "mean_train_mape_5",
            "mean_test_mse",
            "mean_test_mae",
            "mean_test_mse_5",
            "mean_test_mae_5",
            "mean_test_mape_5",
        ]
    ].copy()
    df_new.to_csv("scores_" + names[i] + "_" + str(subset_frac) + ".csv")
