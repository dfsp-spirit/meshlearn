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


##### Settings #####

data_dir = os.path.expanduser("~/develop/cpp_geodesics/output/")

max_files_to_load = 10  # Set to something <= 0 for unlimited.

##### End of settings #####


def load_data(data_dir, max_files_to_load):
    csv_files = glob.glob(os.path.join(data_dir, "*meshdist_edge_5_neigh.csv"))

    num_csv_files = len(csv_files)
    print(f"Discovered {num_csv_files} CSV files containing data in dir '{data_dir}'.")

    if num_csv_files < 1:
        raise ValueError(f"Found no CSV files with training data in directory '{data_dir}'.")

    dataset = pd.read_csv(csv_files[0], sep=" ")

    print(f"Loaded initial dataset with shape {dataset.shape}.")

    ## Read additional CSV files and merge results with 1st one. This may flood RAM for many
    ## large CSV files on less powerful machines. In that case, you will have to use `partial_fit` and train on chunks.
    num_files_loaded = 1
    num_incompatible = 0
    if num_csv_files > 1:
        print(f"Loading additional datasets from the {num_csv_files} files.")
        for idx, filename in enumerate(csv_files):
            if idx == 0:
                print(f" -At file #{idx}: '{filename}': Skipping first one, already loaded")
                continue  # The first one was already loaded.

            if max_files_to_load > 0 and num_files_loaded >= max_files_to_load:   # Limit
                print(f"Loaded {num_files_loaded} CSV files with a total of {len(dataset)} rows so far, ignoring the rest due to 'max_files_to_load'={max_files_to_load} setting.")
                break
            else:   # No limit, keep loading.
                dset_preview = pd.read_csv(filename, sep=" ", nrows=1)
                if len(dset_preview.columns) == len(dataset.columns):
                    dset = pd.read_csv(filename, sep=" ")
                    print(f" -At file #{idx}: Adding {len(dset)} rows from CSV file '{filename}' to current dataset with {len(dataset)} rows.")
                    dataset = pd.concat([dataset, dset], ignore_index=True)
                    num_files_loaded += 1
                else:
                    print(f" -At file #{idx}: Ignoring CSV file '{filename}': it has {len(dset_preview.columns)}, but existing dataset has {len(dataset.columns)}.")
                    num_incompatible += 1

    print(f"Loaded dataset with shape {dataset.shape} from {num_files_loaded} files. Ignored {num_incompatible} CSV files with incompatible column count.")
    return dataset

dataset = load_data(data_dir, max_files_to_load)

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

