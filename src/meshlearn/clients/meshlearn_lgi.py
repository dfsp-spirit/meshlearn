#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
import time
import os
from datetime import timedelta
from sys import getsizeof
import psutil

#from sklearnex import patch_sklearn   # Use Intel extension to speed-up sklearn. Optional, benefits depend on processor type/manufacturer.
#patch_sklearn()                       # Do this BEFORE loading sklearn.


from meshlearn.training_data import get_dataset_pickle
from meshlearn.eval import eval_model_train_test_split, report_feature_importances
from meshlearn.persistance import save_model, load_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def fit_regression_model_sklearnrf(X_train, y_train, model_settings = {'n_estimators':50, 'random_state':0, 'n_jobs': 8}):
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(**model_settings)
    # The 'model_info' is used for a rough overview only. Saved along with pickled model. Not meant for reproduction.
    # Currently needs to be manually adjusted when changing model!
    model_info = {'model_type': 'sklearn.ensemble.RandomForestRegressor', 'model_settings' : model_settings }

    fit_start = time.time()
    regressor.fit(X_train, y_train)

    fit_end = time.time()
    fit_execution_time = fit_end - fit_start
    fit_execution_time_readable = timedelta(seconds=fit_execution_time)
    model_info['fit_time'] = str(fit_execution_time_readable)

    return regressor, model_info

def fit_regression_model_lightgbm(X_train, y_train, model_settings = {'n_estimators':50, 'random_state':0, 'n_jobs':8}):
    from lightgbm import LGBMRegressor
    regressor = LGBMRegressor(**model_settings)
    # The 'model_info' is used for a rough overview only. Saved along with pickled model. Not meant for reproduction.
    # Currently needs to be manually adjusted when changing model!
    model_info = {'model_type': 'lightgbm.LGBMRegressor', 'model_settings' : model_settings }


    fit_start = time.time()
    regressor.fit(X_train, y_train)

    fit_end = time.time()
    fit_execution_time = fit_end - fit_start
    fit_execution_time_readable = timedelta(seconds=fit_execution_time)
    model_info['fit_time'] = str(fit_execution_time_readable)

    return regressor, model_info


"""
Train and evaluate an lGI prediction model.
"""

#default_data_dir = os.path.expanduser("~/data/abide_freesurfer_lgi_2persite")
default_data_dir = "/media/spirit/science/data/abide"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train and evaluate an lGI prediction model.")
parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
parser.add_argument('-d', '--data_dir', help="The recon-all data directory. Created by FreeSurfer.", default=default_data_dir)
parser.add_argument('-n', '--neigh_count', help="Number of vertices to consider at max in the edge neighborhoods for Euclidean dist.", default="500")
parser.add_argument('-r', '--neigh_radius', help="Radius for sphere for Euclidean dist, in spatial units of mesh (e.g., mm).", default="10")
parser.add_argument('-l', '--load_max', help="Total number of samples to load. Set to 0 for all in the files discovered in the data_dir. Used in sequential mode only.", default="0")
parser.add_argument('-p', '--load_per_file', help="Total number of samples to load per file. Set to 0 for all in the respective mesh file.", default="50000")
parser.add_argument('-f', '--load_files', help="Total number of files to load. Set to 0 for all in the data_dir. Used in parallel mode only.", default="48")
parser.add_argument("-s", "--sequential", help="Load data sequentially (as opposed to in parallel, the default).", action="store_true")
parser.add_argument("-c", "--cores", help="Number of cores to use when loading in parallel. Defaults to 0, meaning all.", default="8")
args = parser.parse_args()


# Data settings not exposed on cmd line. Change here if needed.
add_desc_vertex_index = True  # whether to add vertex index as desriptor column to observation
add_desc_neigh_size = True  # whether to add vertex neighborhood size (before pruning) as desriptor column to observation
surface = 'pial'  # The mesh to use.
descriptor = 'pial_lgi'  # The label descriptor, what you want to predict on the mesh.
cortex_label = False  # Whether to load FreeSurfer 'cortex.label' files and filter verts by them. Not implemented yet.
filter_smaller_neighborhoods = False  # Whether to filter (remove) neighborhoods smaller than 'args.neigh_count' (True), or fill the missing columns with 'np.nan' values instead. Note that, if you set to False, you will have to deal with the NAN values in some way before using the data, as most ML models cannot cope with NAN values.


# Construct data settings from command line and other data setting above.
data_settings_in = {'data_dir': args.data_dir, 'surface': surface, 'descriptor' : descriptor, 'cortex_label': cortex_label, 'verbose': args.verbose,
                        'num_neighborhoods_to_load':None if int(args.load_max) == 0 else int(args.load_max), 'num_samples_per_file': None if int(args.load_per_file) == 0 else int(args.load_per_file),
                        'add_desc_vertex_index':add_desc_vertex_index, 'add_desc_neigh_size':add_desc_neigh_size, 'sequential':args.sequential,
                        'num_cores':None if args.cores == "0" else int(args.cores), 'num_files_to_load':None if int(args.load_files) == 0 else int(args.load_files), 'mesh_neighborhood_radius':int(args.neigh_radius),
                        'mesh_neighborhood_count':int(args.neigh_count), 'filter_smaller_neighborhoods': filter_smaller_neighborhoods}

### Other settings, not related to data loading. Adapt here if needed.
do_pickle_data = True
dataset_pickle_file = "ml_dataset.pkl"  # Only relevant if do_pickle_data is True
dataset_settings_file = "ml_dataset.json" # Only relevant if do_pickle_data is True

do_persist_trained_model = True
model_save_file="ml_model.pkl"
model_settings_file="ml_model.json"
num_cores_fit = 8

# Model settings
model_type = "lightgbm"
rf_num_estimators = 48   # For regression problems, take one third of the number of features as a starting point. Also keep your number of cores in mind.
lightgbm_num_estimators = 48


####################################### End of settings. #########################################

print("---Train and evaluate an lGI prediction model---")

if data_settings_in['verbose']:
    print("Verbosity turned on.")

num_cores_tag = "all" if data_settings_in['num_cores'] is None or data_settings_in['num_cores'] == 0 else data_settings_in['num_cores']
seq_par_tag = " sequentially " if data_settings_in['sequential'] else f" in parallel using {num_cores_tag} cores"

if data_settings_in['sequential']:
    print(f"Loading datafiles{seq_par_tag}.")
    print(f"Using data directory '{data_settings_in['data_dir']}', observations to load total limit is set to: {data_settings_in['num_neighborhoods_to_load']}.")
else:
    print("Loading datafiles in parallel.")
    print(f"Using data directory '{data_settings_in['data_dir']}', number of files to load limit is set to: {data_settings_in['num_files_to_load']}.")

print(f"Using neighborhood radius {data_settings_in['mesh_neighborhood_radius']} and keeping {data_settings_in['mesh_neighborhood_count']} vertices per neighborhood.")


print("Descriptor settings:")
if add_desc_vertex_index:
    print(f" - Adding vertex index in mesh as additional descriptor (column) to computed observations (neighborhoods).")
else:
    print(f" - Not adding vertex index in mesh as additional descriptor (column) to computed observations (neighborhoods).")
if add_desc_neigh_size:
    print(f" - Adding neighborhood size before pruning as additional descriptor (column) to computed observations (neighborhoods).")
else:
    print(f" - Not adding neighborhood size before pruning as additional descriptor (column) to computed observations (neighborhoods).")

if data_settings_in['verbose']:
    mem_avail_mb = int(psutil.virtual_memory().available / 1024. / 1024.)
    print(f"RAM available is about {mem_avail_mb} MB.")
    can_estimate = False
    ds_estimated_num_neighborhoods = None
    ds_estimated_num_values_per_neighborhood = 6 * data_settings_in['mesh_neighborhood_count'] + 1
    if data_settings_in['num_neighborhoods_to_load'] is not None and data_settings_in['sequential']:
        # Estimate total dataset size in RAM early to prevent crashing later, if possible.
        ds_estimated_num_neighborhoods = data_settings_in['num_neighborhoods_to_load']
    if data_settings_in['num_samples_per_file'] is not None and data_settings_in['num_files_to_load'] is not None and not data_settings_in['sequential']:
        ds_estimated_num_neighborhoods = data_settings_in['num_samples_per_file'] * data_settings_in['num_files_to_load']
        can_estimate = True
    if can_estimate:
        # try to allocate, will err if too little RAM.
        ds_dummy = np.empty((ds_estimated_num_neighborhoods, ds_estimated_num_values_per_neighborhood))
        ds_estimated_full_data_size_bytes = getsizeof(ds_dummy)
        ds_dummy = None
        ds_estimated_full_data_size_MB = ds_estimated_full_data_size_bytes / 1024. / 1024.
        print(f"Estimated dataset size in RAM will be {int(ds_estimated_full_data_size_MB)} MB.")
        if ds_estimated_full_data_size_MB * 2.0 >= mem_avail_mb:
            print(f"WARNING: Dataset size in RAM is more than half the available memory!") # A simple copy operation will lead to trouble!


dataset, _, data_settings = get_dataset_pickle(data_settings_in, do_pickle_data, dataset_pickle_file, dataset_settings_file)

print(f"Obtained dataset of {int(getsizeof(dataset) / 1024. / 1024.)} MB, containing {dataset.shape[0]} observations, and {dataset.shape[1]} columns ({dataset.shape[1]-1} features + 1 label). {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.")

### NAN handling. Only needed if 'filter_smaller_neighborhoods' is False.
row_indices_with_nan_values = pd.isnull(dataset).any(1).to_numpy().nonzero()[0]
if row_indices_with_nan_values.size > 0:
    print(f"NOTICE: Dataset contains {row_indices_with_nan_values.size} rows (observations) with NAN values (of {dataset.shape[0]} observations total).")
    print(f"NOTICE: You will have to replace these for most models. Set 'filter_smaller_neighborhoods' to 'True' to ignore them when loading data.")
    dataset = dataset.fillna(0, inplace=False) # TODO: replace with something better? Like col mean?
    print(f"Filling NAN values in {row_indices_with_nan_values.size} columns with 0.")
    row_indices_with_nan_values = pd.isnull(dataset).any(1).to_numpy().nonzero()[0]
    print(f"Dataset contains {row_indices_with_nan_values.size} rows (observations) with NAN values (of {dataset.shape[0]} observations total) after filling. {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.")
else:
    print(f"Dataset contains no NAN values. {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.")


nc = len(dataset.columns)
feature_names = np.array(dataset.columns[:-1]) # We require that the label is in the last column of the dataset.
label_name = dataset.columns[-1]
print(f"Separating observations into {len(feature_names)} features and target column '{label_name}'...")

X = dataset.iloc[:, 0:(nc-1)].values
y = dataset.iloc[:, (nc-1)].values

dataset = None

print(f"Splitting data into train and test sets... ({int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X = None # Free RAM.
y = None

print(f"Created training data set with shape {X_train.shape} and testing data set with shape {X_test.shape}. {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.")
print(f"The label arrays have shape {y_train.shape} for the training data and  {y_test.shape} for the testing data.")


print(f"Scaling... (Started at {time.ctime()}, {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.)")

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



if model_type == "sklearnrf":
    print(f"Fitting with RandomForestRegressor with {rf_num_estimators} estimators on {num_cores_fit} cores. (Started at {time.ctime()}.)")
    model_settings_sklearnrf = {'n_estimators':rf_num_estimators, 'random_state':0, 'n_jobs':num_cores_fit}
    model, model_info = fit_regression_model_sklearnrf(X_train, y_train, model_settings = model_settings_sklearnrf)
elif model_type == "lightgbm":
    print(f"Fitting with LightGBM Regressor with {lightgbm_num_estimators} estimators on {num_cores_fit} cores. (Started at {time.ctime()}.)")
    model_settings_lightgbm = {'n_estimators':lightgbm_num_estimators, 'random_state':0, 'n_jobs':num_cores_fit}
    model, model_info = fit_regression_model_lightgbm(X_train, y_train, model_settings=model_settings_lightgbm)
else:
    raise ValueError(f"Invalid model type '{model_type}'.")


model_info = eval_model_train_test_split(model, model_info, X_test, y_test, X_train, y_train)

## Assess feature importance (if possible)
importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
model_info = report_feature_importances(importances, feature_names, model_info)


model_and_data_info = { 'data_settings' : data_settings, 'model_info' : model_info }
if do_persist_trained_model:
    save_model(model, model_and_data_info, model_save_file, model_settings_file)

    ## Some time later, load the model.
    #model, model_and_data_info = load_model(model_save_file, model_settings_file)
    #result = model.score(X_test, Y_test)

