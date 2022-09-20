#!/usr/bin/env python
from __future__ import print_function
from genericpath import isfile
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

from sklearnex import patch_sklearn
patch_sklearn()


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


def get_dataset(data_dir, surface="pial", descriptor="pial_lgi", cortex_label=False, verbose=False, num_neighborhoods_to_load=None, num_samples_per_file=None, add_desc_vertex_index=False, add_desc_neigh_size=False, sequential=True, num_cores=8, num_files_to_load=None):
    discover_start = time.time()
    mesh_files, desc_files, cortex_files, val_subjects, miss_subjects = get_valid_mesh_desc_file_pairs_reconall(data_dir, surface=surface, descriptor=descriptor, cortex_label=cortex_label)
    discover_end = time.time()
    discover_execution_time = discover_end - discover_start
    print(f"=== Discovering data files done, it took: {timedelta(seconds=discover_execution_time)} ===")

    ### Decide which files are used as training, validation and test data. ###
    #input_file_dict = dict(zip(mesh_files, desc_files))  # Dict with mesh_file as key, desc_file as value.
    input_filepair_list = list(zip(mesh_files, desc_files))  # List of 2-tuples, for each tuple first elem is mesh_file, 2nd is desc_file.

    num_cores_tag = "all" if num_cores is None or num_cores == 0 else num_cores
    seq_par_tag = " sequentially " if sequential else f" in parallel using {num_cores_tag} cores"

    if verbose:
        print(f"Discovered {len(input_filepair_list)} valid pairs of input mesh and descriptor files.")

        if sequential:
            if num_neighborhoods_to_load is None:
                print(f"Will load all data from the {len(input_filepair_list)} files{seq_par_tag}.")
            else:
                print(f"Will load {num_neighborhoods_to_load} samples in total from the {len(input_filepair_list)} files.")
        else:
            if num_files_to_load is None:
                print(f"Will load data from all {len(input_filepair_list)} files{seq_par_tag}.")
            else:
                print(f"Will load data from {num_files_to_load} input files.")

        if num_samples_per_file is None:
            print(f"Will load all suitably sized vertex neighborhoods from each mesh file.")
        else:
            print(f"Will load at most {num_samples_per_file} vertex neighborhoods per mesh file.")


    load_start = time.time()
    tdl = TrainingData(neighborhood_radius=mesh_neighborhood_radius, num_neighbors=mesh_neighborhood_count)
    if sequential:
        dataset, col_names = tdl.neighborhoods_from_raw_data_seq(input_filepair_list, num_samples_total=num_neighborhoods_to_load, num_samples_per_file=num_samples_per_file, add_desc_vertex_index=add_desc_vertex_index, add_desc_neigh_size=add_desc_neigh_size)
    else:
        dataset, col_names = tdl.neighborhoods_from_raw_data_parallel(input_filepair_list, num_files_total=num_files_to_load, num_samples_per_file=num_samples_per_file, add_desc_vertex_index=add_desc_vertex_index, add_desc_neigh_size=add_desc_neigh_size, num_cores=num_cores)
    load_end = time.time()
    load_execution_time = load_end - load_start
    print(f"=== Loading data files{seq_par_tag} done, it took: {timedelta(seconds=load_execution_time)} ===")

    assert isinstance(dataset, pd.DataFrame)
    return dataset, col_names



"""
Train and evaluate an lGI prediction model.
"""

#default_data_dir = os.path.expanduser("~/data/abide_freesurfer_lgi_2persite")
default_data_dir = "/media/spirit/science/data/abide"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train and evaluate an lGI prediction model.")
parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
#parser.add_argument('-d', '--data-flat-dir', help="The flat data directory. Use deepcopy_testdata.py script to create.", default="")
parser.add_argument('-d', '--data_dir', help="The recon-all data directory. Created by FreeSurfer.", default=default_data_dir)
parser.add_argument('-n', '--neigh_count', help="Number of vertices to consider at max in the edge neighborhoods for Euclidean dist.", default="300")
parser.add_argument('-r', '--neigh_radius', help="Radius for sphere for Euclidean dist, in spatial units of mesh (e.g., mm).", default="10")
parser.add_argument('-l', '--load_max', help="Total number of samples to load. Set to 0 for all in the files discovered in the data_dir. Used in sequential mode only.", default="500000")
parser.add_argument('-p', '--load_per_file', help="Total number of samples to load per file. Set to 0 for all in the respective mesh file.", default="5000")
parser.add_argument('-f', '--load_files', help="Total number of files to load. Set to 0 for all in the data_dir. Used in parallel mode only.", default="8")
parser.add_argument("-s", "--sequential", help="Load data sequentially (as opposed to in parallel, the default).", action="store_true")
parser.add_argument("-c", "--cores", help="Number of cores to use when loading in parallel. Defaults to 0, meaning all.", default="0")
args = parser.parse_args()

# Post-process the settings from cmd line args (change defaults above if needed)
data_dir = args.data_dir
mesh_neighborhood_count = int(args.neigh_count) # How many vertices in the edge neighborhood do we consider (the 'local' neighbors from which we learn, and from which we take coords, normals, etc as features).
mesh_neighborhood_radius = int(args.neigh_radius)
num_neighborhoods_to_load = None if int(args.load_max) == 0 else int(args.load_max)
num_samples_per_file = None if int(args.load_per_file) == 0 else int(args.load_per_file)
num_files_to_load = None if int(args.load_files) == 0 else int(args.load_files)
sequential = args.sequential
verbose = args.verbose
num_cores = None if args.cores == "0" else int(args.cores)  # Number of cores for loading data in parallel, ignored if sequential is True.

# Other Settings, not exposed on cmd line. Change here if needed.
add_desc_vertex_index = True
add_desc_neigh_size = True

do_pickle_data = True
dataset_pickle_file = "meshlearn_dset.pkl"  # Only relevant if do_pickle_data is True

# Model-specific settings
rf_num_estimators = 50   # For regression problems, take one third of the number of features as a starting point.


print("---Train and evaluate an lGI prediction model---")

if verbose:
    print("Verbosity turned on.")

num_cores_tag = "all" if num_cores is None or num_cores == 0 else num_cores
seq_par_tag = " sequentially " if sequential else f" in parallel using {num_cores_tag} cores"

if sequential:
    print(f"Loading datafiles{seq_par_tag}.")
    print(f"Using data directory '{data_dir}', observations to load total limit is set to: {num_neighborhoods_to_load}.")
else:
    print("Loading datafiles in parallel.")
    print(f"Using data directory '{data_dir}', number of files to load limit is set to: {num_files_to_load}.")

print(f"Using neighborhood radius {mesh_neighborhood_radius} and keeping {mesh_neighborhood_count} vertices per neighborhood.")


print("Descriptor settings:")
if add_desc_vertex_index:
    print(f" - Adding vertex index in mesh as additional descriptor (column) to computed observations (neighborhoods).")
else:
    print(f" - Not adding vertex index in mesh as additional descriptor (column) to computed observations (neighborhoods).")
if add_desc_neigh_size:
    print(f" - Adding neighborhood size before pruning as additional descriptor (column) to computed observations (neighborhoods).")
else:
    print(f" - Not adding neighborhood size before pruning as additional descriptor (column) to computed observations (neighborhoods).")

if num_neighborhoods_to_load is not None and sequential:
    if verbose:
        # Estimate total dataset size in RAM early to prevent crashing later, if possible.
        ds_estimated_num_values_per_neighborhood = 6 * mesh_neighborhood_count + 1
        ds_estimated_num_neighborhoods = num_neighborhoods_to_load
        # try to allocate, will err if too little RAM.
        mem_avail_mb = int(psutil.virtual_memory().available / 1024. / 1024.)
        print(f"RAM available is about {mem_avail_mb} MB")
        ds_dummy = np.empty((ds_estimated_num_neighborhoods, ds_estimated_num_values_per_neighborhood))
        ds_estimated_full_data_size_bytes = getsizeof(ds_dummy)
        ds_dummy = None
        ds_estimated_full_data_size_MB = ds_estimated_full_data_size_bytes / 1024. / 1024.
        print(f"Estimated dataset size in RAM will be {int(ds_estimated_full_data_size_MB)} MB.")
        if ds_estimated_full_data_size_MB * 2.0 >= mem_avail_mb:
            print(f"WARNING: Dataset size in RAM is more than half the available memory!") # A simple copy operation will lead to trouble!


if do_pickle_data and os.path.isfile(dataset_pickle_file):
    print(f"WARNING: Unpickling pre-saved dataframe from pickle file '{dataset_pickle_file}', ignoring all settings! Delete file or set 'do_pickle_data'to False to prevent.")
    unpickle_start = time.time()
    dataset = pd.read_pickle(dataset_pickle_file)
    col_names = dataset.columns
    unpickle_end = time.time()
    pickle_load_time = unpickle_end - unpickle_start
    print(f"INFO: Loaded dataset with shape {dataset.shape} from pickle file '{dataset_pickle_file}'. It took  {timedelta(seconds=pickle_load_time)}.")
else:
    dataset, col_names = get_dataset(data_dir, surface="pial", descriptor="pial_lgi", cortex_label=False, verbose=verbose, num_neighborhoods_to_load=num_neighborhoods_to_load, num_samples_per_file=num_samples_per_file, add_desc_vertex_index=add_desc_vertex_index, add_desc_neigh_size=add_desc_neigh_size, sequential=sequential, num_cores=num_cores, num_files_to_load=num_files_to_load)
    if do_pickle_data:
        pickle_start = time.time()
        dataset.to_pickle(dataset_pickle_file)
        pickle_end = time.time()
        pickle_save_time = pickle_end - pickle_start
        print(f"INFO: Saved dataset to pickle file '{dataset_pickle_file}', ready to load next run. It took  {timedelta(seconds=pickle_save_time)}.")


print(f"Obtained dataset of  {int(getsizeof(dataset) / 1024. / 1024.)} MB, containing {dataset.shape[0]} observations, and {dataset.shape[1]} columns ({dataset.shape[1]-1}features + 1 label). {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.")
print("Separating observations and labels...")

nc = len(dataset.columns)
X = dataset.iloc[:, 0:(nc-1)].values
y = dataset.iloc[:, (nc-1)].values

print("Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"Created training data set with shape {X_train.shape} and testing data set with shape {X_test.shape}.")
print(f"The label arrays have shape {y_train.shape} for the training data and  {y_test.shape} for the testing data.")


print("Scaling...")

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


fit_start = time.time()
print(f"Fitting with RandomForestRegressor with {rf_num_estimators} estimators. (Started at {time.ctime()}.)")

regressor = RandomForestRegressor(n_estimators=rf_num_estimators, random_state=0, n_jobs=-1)
regressor.fit(X_train, y_train)

fit_end = time.time()
fit_execution_time = fit_end - fit_start

print(f"===Fitting done, it took: {timedelta(seconds=fit_execution_time)} ===")
print(f"Using trained model to predict for test data set with shape {X_test.shape}.")

y_pred = regressor.predict(X_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Evaluate feature importance
importances = regressor.feature_importances_

feature_names = np.array(col_names[:-1])
assert len(feature_names) == len(importances)

print(f"=== Evaluating Feature importance ===")
print(f"Feature names       : {feature_names}")
print(f"Feature importances : {importances}")

max_importance = np.max(importances)
min_importance = np.min(importances)
mean_importance = np.mean(importances)

print(f"Max feature importance is {max_importance}, min is {min_importance}, mean is {mean_importance}.")
max_important_idx = np.argmax(importances)
min_important_idx = np.argmin(importances)
print(f"Most important feature is {feature_names[max_important_idx]}, min important one is {feature_names[min_important_idx]}.")

sorted_indices = np.argsort(importances)
num_to_report = int(min(10, len(feature_names)))
print(f"Most important {num_to_report} features are: {feature_names[sorted_indices[-num_to_report:]]}")
print(f"Least important {num_to_report} features are: {feature_names[sorted_indices[0:num_to_report]]}")

do_plot = False
if do_plot:
    std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()


