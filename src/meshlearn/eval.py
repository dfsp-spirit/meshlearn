# -*- coding: utf-8 -*-

"""
Model evaluation functions.
"""

import numpy as np
from sklearn import metrics

def eval_model_train_test_split(model, model_info, X_test, y_test, X_train, y_train, X_eval=None, y_eval=None):
    """
    Evaluate a model based on train/test split (no CV).
    """
    print(f"Using trained model to predict for test data set with shape {X_test.shape}.")

    if model_info is None:
        model_info = dict()

    y_test_pred = model.predict(X_test)

    print('Performance on test data (for model evaluation)')
    mae_test = metrics.mean_absolute_error(y_test, y_test_pred)
    mse_test = metrics.mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    print(f' - Mean Absolute Error    : {mae_test}')
    print(f' - Mean Squared Error     : {mse_test}')
    print(f' - Root Mean Squared Error: {rmse_test}')
    if not 'evaluation' in model_info:
        model_info['evaluation'] = dict()
    model_info['evaluation']['mae_test'] = mae_test.item()
    model_info['evaluation']['mse_test'] = mse_test.item()
    model_info['evaluation']['rmse_test'] = rmse_test.item()

    if X_eval is not None and y_eval is not None:
        y_eval_pred = model.predict(X_eval)
        print('Performance on evaluation set (not for model evaluation: eval has been used for cross-validation)')
        mae_eval = metrics.mean_absolute_error(y_eval, y_eval_pred)
        mse_eval = metrics.mean_squared_error(y_eval, y_eval_pred)
        rmse_eval = np.sqrt(metrics.mean_squared_error(y_eval, y_eval_pred))
        print(f' - Mean Absolute Error on eval data (Used for hyperparameter optimization. Do not use for model evaluation!)    : {mae_eval}')
        print(f' - Mean Squared Error on eval data (Used for hyperparameter optimization. Do not use for model evaluation!)     : {mse_eval}')
        print(f' - Root Mean Squared Error on eval data (Used for hyperparameter optimization. Do not use for model evaluation!): {rmse_eval}')
        if not 'evaluation' in model_info:
            model_info['evaluation'] = dict()
        if not 'extra_info' in model_info['evaluation']:
            model_info['evaluation']['extra_info'] = dict()
        model_info['evaluation']['extra_info']['mae_eval'] = mae_eval.item()
        model_info['evaluation']['extra_info']['mse_eval'] = mse_eval.item()
        model_info['evaluation']['extra_info']['rmse_eval'] = rmse_eval.item()

    print('Performance on training data (for underfitting/overfitting estimation only)')
    y_train_pred = model.predict(X_train) # Fits on training data the model has already seen!
    mae_train = metrics.mean_absolute_error(y_train, y_train_pred)
    mse_train = metrics.mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    if not 'evaluation' in model_info:
            model_info['evaluation'] = dict()
    if not 'extra_info' in model_info['evaluation']:
        model_info['evaluation']['extra_info'] = dict()
    model_info['evaluation']['extra_info']['mae_train'] = mae_train.item()
    model_info['evaluation']['extra_info']['mse_train'] = mse_train.item()
    model_info['evaluation']['extra_info']['rmse_train'] = rmse_train.item()
    print(f' - Mean Absolute Error on training data (Used for fitting. Do not use for model evaluation!)     : {mae_train}')
    print(f' - Mean Squared Error on training data (Used for fitting. Do not use for model evaluation!)      : {mse_train}')
    print(f' - Root Mean Squared Error on training data (Used for fitting. Do not use for model evaluation!) : {rmse_train}')

    return model_info


def report_feature_importances(importances, feature_names, model_info=None, num_to_report=20):
    if model_info is None:
        model_info = dict()
    if importances is not None: # Some models do not support it.
        assert len(feature_names) == len(importances)

        print(f"=== Evaluating Feature importance ===")
        print(f"Feature names       : {feature_names}")
        print(f"Feature importances : {importances}")

        max_importance = np.max(importances)
        min_importance = np.min(importances)
        mean_importance = np.mean(importances)
        median_importance = np.median(importances)

        print(f"Max feature importance is {max_importance}, min is {min_importance}, mean is {mean_importance}, median is {median_importance}.")
        max_important_idx = np.argmax(importances)
        min_important_idx = np.argmin(importances)
        print(f"Most important feature is {feature_names[max_important_idx]}, min important one is {feature_names[min_important_idx]}.")

        if not 'feature_importances' in model_info:
            model_info['feature_importances'] = dict()

        model_info['feature_importances']['max_importance_score'] = max_importance.item() # Cannot serialize np.float
        model_info['feature_importances']['min_importance_score'] = min_importance.item()
        model_info['feature_importances']['mean_importance_score'] = mean_importance.item()
        model_info['feature_importances']['median_importance_score'] = median_importance.item()

        sorted_indices = np.argsort(importances)
        num_to_report = int(min(num_to_report, len(feature_names)))
        most_important_features_names = feature_names[sorted_indices[-num_to_report:]]
        most_important_features_importances = importances[sorted_indices[-num_to_report:]]
        least_important_features_names = feature_names[sorted_indices[0:num_to_report]]
        least_important_features_importances = importances[sorted_indices[0:num_to_report]]
        print(f"The {num_to_report} most important {num_to_report} features are: {most_important_features_names}")
        print(f" * Their importances are: {most_important_features_importances}")
        print(f"The {num_to_report} least important {num_to_report} features are: {least_important_features_names}")
        print(f" * Their importances are: {least_important_features_importances}")
        model_info['feature_importances']['most_important_features_names'] = most_important_features_names.tolist() # Cannot serialize np.ndarry to JSON.
        model_info['feature_importances']['most_important_features_importances'] = most_important_features_importances.tolist()
        model_info['feature_importances']['least_important_features_names'] = least_important_features_names.tolist()
        model_info['feature_importances']['least_important_features_importances'] = least_important_features_importances.tolist()
    else:
        print(f"Cannot evaluate feature importance for model, None supplied.")
    return model_info


