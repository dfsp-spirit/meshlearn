#!/usr/bin/env python
import json
import numpy as np
import argparse
import time
from datetime import timedelta
from sys import getsizeof
import psutil

from sklearnex import patch_sklearn   # Use Intel extension to speed-up sklearn. Optional, benefits depend on processor type/manufacturer.
patch_sklearn()                       # Do this BEFORE loading sklearn.


from meshlearn.training_data import get_dataset_pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def fit_regression_model_sklearnrf(X_train, y_train, model_settings = {'n_estimators':50, 'random_state':0, 'n_jobs': 8}):
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(**model_settings)
    # The 'model_info' is used for a rough overview only. Saved along with pickled model. Not meant for reproduction.
    # Currently needs to be manually adjusted when changing model!
    model_info = {'model_type': 'sklearn.ensemble.RandomForestRegressor', 'model_settings' : model_settings }


    regressor.fit(X_train, y_train)

    fit_end = time.time()
    fit_execution_time = fit_end - fit_start
    fit_execution_time_readable = timedelta(seconds=fit_execution_time)
    model_info['fit_time'] = str(fit_execution_time_readable)

    print(f"===Fitting done, it took: {fit_execution_time_readable} ===")
    print(f"Using trained model to predict for test data set with shape {X_test.shape}.")

    y_pred = regressor.predict(X_test)


    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # Evaluate feature importance
    importances = regressor.feature_importances_
    return regressor, model_info, importances

def fit_regression_model_lightgbm(X_train, y_train, model_settings = {'n_estimators':50, 'random_state':0, 'n_jobs':8}):
    from lightgbm import LGBMRegressor
    regressor = LGBMRegressor()
    # The 'model_info' is used for a rough overview only. Saved along with pickled model. Not meant for reproduction.
    # Currently needs to be manually adjusted when changing model!
    model_info = {'model_type': 'lightgbm.LGBMRegressor', 'model_settings' : model_settings }


    regressor.fit(X_train, y_train)

    fit_end = time.time()
    fit_execution_time = fit_end - fit_start
    fit_execution_time_readable = timedelta(seconds=fit_execution_time)
    model_info['fit_time'] = str(fit_execution_time_readable)

    print(f"===Fitting done, it took: {fit_execution_time_readable} ===")
    print(f"Using trained model to predict for test data set with shape {X_test.shape}.")

    y_pred = regressor.predict(X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # Evaluate feature importance
    importances = regressor.feature_importances_
    return regressor, model_info, importances


"""
Train and evaluate an lGI prediction model.
"""

#default_data_dir = os.path.expanduser("~/data/abide_freesurfer_lgi_2persite")
default_data_dir = "/media/spirit/science/data/abide"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train and evaluate an lGI prediction model.")
parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
parser.add_argument('-d', '--data_dir', help="The recon-all data directory. Created by FreeSurfer.", default=default_data_dir)
parser.add_argument('-n', '--neigh_count', help="Number of vertices to consider at max in the edge neighborhoods for Euclidean dist.", default="300")
parser.add_argument('-r', '--neigh_radius', help="Radius for sphere for Euclidean dist, in spatial units of mesh (e.g., mm).", default="10")
parser.add_argument('-l', '--load_max', help="Total number of samples to load. Set to 0 for all in the files discovered in the data_dir. Used in sequential mode only.", default="0")
parser.add_argument('-p', '--load_per_file', help="Total number of samples to load per file. Set to 0 for all in the respective mesh file.", default="50000")
parser.add_argument('-f', '--load_files', help="Total number of files to load. Set to 0 for all in the data_dir. Used in parallel mode only.", default="40")
parser.add_argument("-s", "--sequential", help="Load data sequentially (as opposed to in parallel, the default).", action="store_true")
parser.add_argument("-c", "--cores", help="Number of cores to use when loading in parallel. Defaults to 0, meaning all.", default="0")
args = parser.parse_args()


# Data settings not exposed on cmd line. Change here if needed.
add_desc_vertex_index = True  # whether to add vertex index as desriptor column to observation
add_desc_neigh_size = True  # whether to add vertex neighborhood size (before pruning) as desriptor column to observation
surface = 'pial' # The mesh to use.
descriptor = 'pial_lgi' # The label descriptor, what you want to predict on the mesh.
cortex_label = False # Whether to load FreeSurfer 'cortex.label' files and filter verts by them. Not implemented yet.

# Construct data settings from command line and other data setting above.
data_settings_in = {'data_dir': args.data_dir, 'surface': surface, 'descriptor' : descriptor, 'cortex_label': cortex_label, 'verbose': args.verbose,
                        'num_neighborhoods_to_load':None if int(args.load_max) == 0 else int(args.load_max), 'num_samples_per_file': None if int(args.load_per_file) == 0 else int(args.load_per_file),
                        'add_desc_vertex_index':add_desc_vertex_index, 'add_desc_neigh_size':add_desc_neigh_size, 'sequential':args.sequential,
                        'num_cores':None if args.cores == "0" else int(args.cores), 'num_files_to_load':None if int(args.load_files) == 0 else int(args.load_files), 'mesh_neighborhood_radius':int(args.neigh_radius),
                        'mesh_neighborhood_count':int(args.neigh_count)}

### Other settings, not related to data loading. Adapt here if needed.
do_pickle_data = True
dataset_pickle_file = "meshlearn_dset.pkl"  # Only relevant if do_pickle_data is True
dataset_settings_file = "meshlearn_dset_settings.json" # Only relevant if do_pickle_data is True

do_persist_trained_model = True
model_save_file="model.pkl"
model_settings_file="model_settings.json"
num_cores_fit = 8


### Model-specific settings
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

if data_settings_in['num_neighborhoods_to_load'] is not None and data_settings_in['sequential']:
    if data_settings_in['verbose']:
        # Estimate total dataset size in RAM early to prevent crashing later, if possible.
        ds_estimated_num_values_per_neighborhood = 6 * data_settings_in['mesh_neighborhood_count'] + 1
        ds_estimated_num_neighborhoods = data_settings_in['num_neighborhoods_to_load']
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


dataset, col_names, data_settings = get_dataset_pickle(data_settings_in, do_pickle_data, dataset_pickle_file, dataset_settings_file)

print(f"Obtained dataset of  {int(getsizeof(dataset) / 1024. / 1024.)} MB, containing {dataset.shape[0]} observations, and {dataset.shape[1]} columns ({dataset.shape[1]-1} features + 1 label). {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.")
print("Separating observations and labels...")

nc = len(dataset.columns)
X = dataset.iloc[:, 0:(nc-1)].values
y = dataset.iloc[:, (nc-1)].values

print("Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"Created training data set with shape {X_train.shape} and testing data set with shape {X_test.shape}.")
print(f"The label arrays have shape {y_train.shape} for the training data and  {y_test.shape} for the testing data.")


print(f"Scaling... (Started at {time.ctime()}.)")

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




fit_start = time.time()
#print(f"Fitting with RandomForestRegressor with {rf_num_estimators} estimators on {num_cores_fit} cores. (Started at {time.ctime()}.)")
#regressor, model_info, importances = fit_regression_model_sklearnrf(X_train, y_train, model_settings = {'n_estimators':rf_num_estimators, 'random_state':0, 'n_jobs':num_cores_fit})

print(f"Fitting with LightGBM Regressor with {lightgbm_num_estimators} estimators on {num_cores_fit} cores. (Started at {time.ctime()}.)")
regressor, model_info, importances = fit_regression_model_lightgbm(X_train, y_train, model_settings = {'n_estimators':lightgbm_num_estimators, 'random_state':0, 'n_jobs':num_cores_fit})

## Assess feature importance (if possible)
if importances is not None: # Some regressors do not support it.
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


if do_persist_trained_model:
    import pickle
    # save the model to disk
    pickle_model_start = time.time()
    pickle.dump(regressor, open(model_save_file, 'wb'))
    # Save the model settings as a JSON file.
    model_and_data_settings = { 'data_settings' : data_settings, 'model_info' : model_info }
    with open(model_settings_file, 'w') as fp:
        json.dump(model_and_data_settings, fp, sort_keys=True, indent=4)


    pickle_model_end = time.time()
    pickle_model_save_time = pickle_model_end - pickle_model_start
    print(f"INFO: Saved trained model to pickle file '{model_save_file}', ready to load later. Saving model took {timedelta(seconds=pickle_model_save_time)}.")

    ## Some time later, load 'model.pkl'
    #loaded_model = pickle.load(open(model_save_file, 'rb'))
    #result = loaded_model.score(X_test, Y_test)

