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


from sklearn.model_selection import train_test_split
from meshlearn.tfdata import VertexPropertyDataset
from meshlearn.training_data import TrainingData, get_valid_mesh_desc_lgi_file_pairs

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# PYTHONPATH=./src/meshlearn python src/meshlearn/clients/meshlearn_lgi.py --verbose

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics




def meshlearn_lgi():
    """
    Train and evaluate an lGI prediction model.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate an lGI prediction model.")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    parser.add_argument('-d', '--data-dir', help="The data directory. Use deepcopy_testdata.py script to create.", default="./tests/test_data/tim_only")
    parser.add_argument('-e', '--epochs', help="Number of training epochs.", default="20")
    parser.add_argument('-n', '--neigh_count', help="Number of vertices to consider at max in the edge neighborhoods for Euclidean dist.", default="100")
    parser.add_argument('-r', '--neigh_radius', help="Radius for sphere for Euclidean dist, in spatial units of mesh (e.g., mm).", default="10")
    args = parser.parse_args()

    num_epochs = int(args.epochs)
    data_dir = args.data_dir
    mesh_neighborhood_count = int(args.neigh_count) # How many vertices in the edge neighborhood do we consider (the 'local' neighbors from which we learn).
    mesh_neighborhood_radius = int(args.neigh_radius)


    print("---Train and evaluate an lGI prediction model---")
    if args.verbose:
        print("Verbosity turned on.")
        print("Training for {num_epochs} epochs.".format(num_epochs=num_epochs))
        print("Using data directory '{data_dir}'.".format(data_dir=data_dir))

    mesh_files, desc_files = get_valid_mesh_desc_lgi_file_pairs(data_dir, args.verbose)

    ### Decide which files are used as training, validation and test data. ###
    input_file_dict = dict(zip(mesh_files, desc_files))

    if args.verbose:
        print(f"Discovered {len(input_file_dict)} valid pairs of input mesh and descriptor files.")

    num_neighborhoods_to_load = 50000

    tdl = TrainingData(neighborhood_radius=mesh_neighborhood_radius, num_neighbors=mesh_neighborhood_count)
    dataset = tdl.load_raw_data(input_file_dict, num_samples_to_load=num_neighborhoods_to_load)

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



    n_estimators = 20
    print(f"Fitting with RandomForestRegressor with {n_estimators} estimators.")

    fit_start = time.time()

    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    regressor.fit(X_train, y_train)

    fit_end = time.time()
    execution_time = fit_end - fit_start

    print(f"Fitting done, it took: {timedelta(seconds=execution_time)}")
    print(f"Using trained model to predict for test data set with shape {X_test.shape}.")

    y_pred = regressor.predict(X_test)


    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


if __name__ == "__main__":
    meshlearn_lgi()
