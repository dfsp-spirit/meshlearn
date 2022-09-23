# -*- coding: utf-8 -*-

"""
Model evaluation functions.
"""

import numpy as np
from sklearn import metrics

def eval_model_train_test_split(model, model_info, X_test, y_test, X_train, y_train):
    """
    Evaluate a model based on train/test split (no CV).
    """
    print(f"Using trained model to predict for test data set with shape {X_test.shape}.")

    y_pred = model.predict(X_test)

    print('Performance on test data (for model evaluation)')
    mae_test = metrics.mean_absolute_error(y_test, y_pred)
    mse_test = metrics.mean_squared_error(y_test, y_pred)
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(f' - Mean Absolute Error    : {mae_test}')
    print(f' - Mean Squared Error     : {mse_test}')
    print(f' - Root Mean Squared Error: {rmse_test}')
    model_info['evaluation'] = dict()
    model_info['evaluation']['mae_test'] = mae_test
    model_info['evaluation']['mse_test'] = mse_test
    model_info['evaluation']['rmse_test'] = rmse_test

    print('Performance on training data (for underfitting/overfitting estimation only)')
    y_train_pred = model.predict(X_train) # Fits on training data the model has already seen!
    mae_train = metrics.mean_absolute_error(y_train, y_train_pred)
    mse_train = metrics.mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    model_info['evaluation']['mae_train'] = mae_train
    model_info['evaluation']['mse_train'] = mse_train
    model_info['evaluation']['rmse_train'] = rmse_train
    print(f' - Mean Absolute Error on training data (Do not use for model evaluation!)     : {mae_train}')
    print(f' - Mean Squared Error on training data (Do not use for model evaluation!)      : {mse_train}')
    print(f' - Root Mean Squared Error on training data (Do not use for model evaluation!) : {rmse_train}')

    return model, model_info

