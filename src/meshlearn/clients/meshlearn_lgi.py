#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import time
from datetime import timedelta
import psutil


from sklearn.model_selection import train_test_split
from meshlearn.tfdata import VertexPropertyDataset
from meshlearn.training_data import TrainingData, get_valid_mesh_desc_lgi_file_pairs, get_valid_mesh_desc_file_pairs_reconall

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# PYTHONPATH=./src/meshlearn python src/meshlearn/clients/meshlearn_lgi.py --verbose

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sys import getsizeof





"""
Train and evaluate an lGI prediction model.
"""

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train and evaluate an lGI prediction model.")
parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
#parser.add_argument('-d', '--data-flat-dir', help="The flat data directory. Use deepcopy_testdata.py script to create.", default="")
parser.add_argument('-d', '--data_dir', help="The recon-all data directory. Created by FreeSurfer.", default="/media/spirit/science/data/abide")
parser.add_argument('-n', '--neigh_count', help="Number of vertices to consider at max in the edge neighborhoods for Euclidean dist.", default="300")
parser.add_argument('-r', '--neigh_radius', help="Radius for sphere for Euclidean dist, in spatial units of mesh (e.g., mm).", default="10")
parser.add_argument('-l', '--load_max', help="Number of samples to load. Set to 0 for all in the files discovered in the data_dir.", default="400000")
args = parser.parse_args()

data_dir = args.data_dir
mesh_neighborhood_count = int(args.neigh_count) # How many vertices in the edge neighborhood do we consider (the 'local' neighbors from which we learn).
mesh_neighborhood_radius = int(args.neigh_radius)

num_neighborhoods_to_load = None if int(args.load_max) == 0 else int(args.load_max)

print("---Train and evaluate an lGI prediction model---")
if args.verbose:
    print("Verbosity turned on.")

print(f"Using data directory '{data_dir}', observations to load limit is set to: {num_neighborhoods_to_load}.")
print(f"Using neighborhood radius {mesh_neighborhood_radius} and keeping {mesh_neighborhood_count} vertices per neighborhood.")

if num_neighborhoods_to_load is not None:
    # Estimate total dataset size in RAM early to prevent crashing later, if possible.
    ds_estimated_num_values_per_neighborhood = 6 * mesh_neighborhood_count + 1
    ds_estimated_num_neighborhoods = num_neighborhoods_to_load
    # try to allocate, will err if too little RAM.
    print(f"RAM available is about {int(psutil.virtual_memory().available / 1024. / 1024.)} MB")
    ds_dummy = np.empty((ds_estimated_num_neighborhoods, ds_estimated_num_values_per_neighborhood))
    ds_estimated_full_data_size_bytes = getsizeof(ds_dummy)
    ds_dummy = None
    ds_estimated_full_data_size_MB = ds_estimated_full_data_size_bytes / 1024. / 1024.
    print(f"Estimated dataset size in RAM will be {int(ds_estimated_full_data_size_MB)} MB.")



discover_start = time.time()
mesh_files, desc_files, cortex_files, val_subjects = get_valid_mesh_desc_file_pairs_reconall(data_dir)
discover_end = time.time()
discover_execution_time = discover_end - discover_start
print(f"Discovering data files done, it took: {timedelta(seconds=discover_execution_time)}")

### Decide which files are used as training, validation and test data. ###
input_file_dict = dict(zip(mesh_files, desc_files))

if args.verbose:
    print(f"Discovered {len(input_file_dict)} valid pairs of input mesh and descriptor files.")

if num_neighborhoods_to_load is None:
    print(f"Will load all data from the {len(input_file_dict)} files.")
else:
    print(f"Will load {num_neighborhoods_to_load} samples in total from the {len(input_file_dict)} files.")

load_start = time.time()
tdl = TrainingData(neighborhood_radius=mesh_neighborhood_radius, num_neighbors=mesh_neighborhood_count)
dataset = tdl.load_raw_data(input_file_dict, num_samples_to_load=num_neighborhoods_to_load)
load_end = time.time()
load_execution_time = load_end - load_start
print(f"Loading data files done, it took: {timedelta(seconds=load_execution_time)}")

assert isinstance(dataset, pd.DataFrame)

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

n_estimators = 100
print(f"Fitting with RandomForestRegressor with {n_estimators} estimators.")

fit_start = time.time()

regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=-1)
regressor.fit(X_train, y_train)

fit_end = time.time()
fit_execution_time = fit_end - fit_start

print(f"Fitting done, it took: {timedelta(seconds=fit_execution_time)}")
print(f"Using trained model to predict for test data set with shape {X_test.shape}.")

y_pred = regressor.predict(X_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
