#!/usr/bin/env python
from __future__ import print_function

import pandas as pd
import numpy as np
import os.path
import glob
import time
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from meshlearn.load_preproc_data import load_preproc_csv_data

##### Settings #####

data_dir = os.path.expanduser("~/develop/cpp_geodesics/output/")

max_files_to_load = 10  # Set to something <= 0 for unlimited.

##### End of settings #####


dataset = load_preproc_csv_data(data_dir, max_files_to_load)

nc = len(dataset.columns)
X = dataset.iloc[:, 0:(nc-1)].values
y = dataset.iloc[:, (nc-1)].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"Created training data set with shape {X_train.shape} and testing data set with shape {X_test.shape}.")
print(f"The label arrays have shape {y_train.shape} for the training data and  {y_test.shape} for the testing data.")


print("Scaling...")

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



n_estimators = 20
print(f"Fitting with RandomForestRegressor with {n_estimators} estimators.")

fit_start = time.time()

regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
regressor.fit(X_train, y_train)

fit_end = time.time()
execution_time = fit_end - fit_start

print(f"Fitting done, it took: {timedelta(seconds=execution_time)}")
print("Using trained model to predict for test data set with shape {X_test.shape}.")

y_pred = regressor.predict(X_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

