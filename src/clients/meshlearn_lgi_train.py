#!/usr/bin/env python

"""
This program trains a LightGBM model on local gyrification index (lGI) per-vertex data.
"""

import numpy as np
import pandas as pd
import argparse
import os
import time
from datetime import timedelta
from sys import getsizeof
import psutil
import lightgbm
import gc  # Garbage collection.


#from sklearnex import patch_sklearn   # Use Intel extension to speed-up sklearn. Optional, benefits depend on processor type/manufacturer.
#patch_sklearn()                       # Do this BEFORE loading sklearn.

import matplotlib.pyplot as plt
plt.ion()

from meshlearn.data.training_data import get_dataset_pickle
from meshlearn.model.eval import eval_model_train_test_split, report_feature_importances
from meshlearn.model.persistance import save_model, load_model
from meshlearn.data.postproc import postproc_settings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def fit_regression_model_lightgbm(X_train, y_train, X_val, y_val,
                                  model_settings = {'n_estimators':50, 'random_state':42, 'n_jobs':8},
                                  opt_fit_settings = {'colsample_bytree': 0.8532168461905915, 'min_child_samples': 489, 'min_child_weight': 10.0,
                                                      'num_leaves': 47, 'reg_alpha': 2, 'reg_lambda': 20, 'subsample': 0.22505063396444688} ):
    """Fit a lightgbm model with hard-coded parameters. Pass params obtained via `hyperparameter_optimization_lightgbm` in an earlier run as `opt_fit_settings`."""
    regressor = lightgbm.LGBMRegressor(**model_settings)

    # The `opt_fit_settings` were computed using `hyperparameter_optimization_lightgbm`.
    regressor.set_params(**opt_fit_settings)
    # The 'model_info' is used for a rough overview only. Saved along with pickled model. Not meant for reproduction.
    # Currently needs to be manually adjusted when changing model!
    model_info = {'model_type': 'lightgbm.LGBMRegressor', 'model_settings' : model_settings }

    fit_start = time.time()
    regressor.fit(X_train, y_train, eval_set=[(X_val, y_val), (X_train, y_train)], eval_names=["validation set", "training set"])

    fit_end = time.time()
    fit_execution_time = fit_end - fit_start
    fit_execution_time_readable = timedelta(seconds=fit_execution_time)
    model_info['fit_time'] = str(fit_execution_time_readable)

    return regressor, model_info


def hyperparameter_optimization_lightgbm(X_train, y_train, X_val, y_val, num_iterations = 20, inner_cv_k=3, hpt_num_estimators=100, num_cores=8, random_state=42, eval_metric="neg_mean_absolute_error", verbose_lightgbm=1, verbose_random_search=2, return_opt_model=True):
    """Perform hypermarameter optimization via random parameter search.

    This takes (num_iterations x inner_cv_k) times longer than fitting a single model. Run this once, hard-code the obtained
    parameters, and from then on run `fit_regression_model_lightgbm` with these parameters instead.

    Parameters
    ----------
    ...
    return_opt_model : bool, whether to re-fit a final model on all data with the optimal params and return it. If `False`, the `model` return value will be `None`.

    Returns
    -------
    model: None or lightgbm.model instance, depending on parameter `return_opt_model`
    model_info : dict with subdicts `model_params` and `fit_params`, where `model_params` are the tuned, optimal params. The `fit_params` are other settings used for the model fit that were fixed during optimization.
    """

    # Credits: was based https://www.kaggle.com/code/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761
    print(f'Performing hyperparameter optimization for lightgbm Model using {num_iterations} random search iterations and {inner_cv_k}-fold inner cross validation ({num_iterations * inner_cv_k} fits total) using {num_cores} cores.')

    model_info = {'model_type': 'lightgbm.LGBMRegressor'}

    from scipy.stats import randint as sp_randint
    from scipy.stats import uniform as sp_uniform

    def learning_rate_010_decay_power_0995(current_iter):
        """Callback function for learning rate decay."""
        base_learning_rate = 0.1
        lr = base_learning_rate  * np.power(.995, current_iter)
        return lr if lr > 1e-3 else 1e-3

    fit_params = { "eval_metric" : eval_metric,
                    "eval_set" : [(X_val, y_val)],
                    "eval_names" : ["validation set"]
                    #"callbacks" : [
                    #                lightgbm.reset_parameter(learning_rate=learning_rate_010_decay_power_0995),
                    #                lightgbm.early_stopping(stopping_rounds=20, verbose = verbose),
                    #              ]
                 }

    param_test = { 'num_leaves': sp_randint(30, 100),
                   'min_child_samples': sp_randint(100, 500),
                   'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                   'subsample': sp_uniform(loc=0.2, scale=0.8),
                   'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                   'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                   'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100] }


    from sklearn.model_selection import RandomizedSearchCV

    # Note: n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and
    # the large value 5000 defines only the absolute maximum, that will not be reached in practice.
    model_param_search = lightgbm.LGBMRegressor(random_state=random_state, verbose=verbose_lightgbm, n_jobs=num_cores, n_estimators=hpt_num_estimators)
    random_search = RandomizedSearchCV(
        estimator=model_param_search,
        param_distributions=param_test,
        n_iter=num_iterations,
        scoring=eval_metric,
        cv=inner_cv_k,
        refit=False,
        random_state=random_state,
        verbose=verbose_random_search)

    print(f"[hyperparam_opt] Running random search with fit_params: '{fit_params}'")

    random_search.fit(X_train, y_train, **fit_params)
    opt_params = random_search.best_params_
    print(f'Best score reached: {random_search.best_score_} with params: {opt_params}. {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.')

    model = None
    if return_opt_model:
        print(f'Fitting final model...')
        model = lightgbm.LGBMRegressor(**model_param_search.get_params())
        #set optimal parameters
        model.set_params(**opt_params)
        model.fit(X_train, y_train, **fit_params)
    model_info['model_params'] =  opt_params
    model_info['fit_params'] =  fit_params
    return model, model_info


def train_lgi():
    """
    Train and evaluate an lGI prediction model.
    """

    # Parse command line arguments
    example_text = '''Examples:

 # Use neighborhood radius 15mm, and coords/normals for max 300 vertices in radius, load 500k vertex neighborhoods:
 meshlearn_lgi_train -v -n 300 -r 15 -l 500000 -s -c 4 $SUBJECTS_DIR
 # Use neighborhood defaults, load 35k samples per file from 48 files, 8 in parallel. Persist dataset before training:
 meshlearn_lgi_train -p 35000 -f 48 -c 8 -t "_v2" -w . $SUBJECTS_DIR
 # After running the previous command, load the persisted dataset for training new model (with changed training settings in script):
 meshlearn_lgi_train -t "_v2" -w . $SUBJECTS_DIR

 Note: The dataset to be loaded must fit into RAM. To get great results, we suggest
 running model training on a machine with at least 128 GB of RAM, and to load a diverse
 set of training files (-p) which amount to 50 - 60 GB of data im memory. Use slighly less
 than half of the available RAM for the data to load (many ops need to temporarily store
 another copy of the data in RAM).
 '''
    parser = argparse.ArgumentParser(prog='meshlearn_lgi_train',
                                     description="Train and evaluate an lGI prediction model.",
                                     epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter
                                     )

    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    parser.add_argument('data_dir', help="The recon-all data directory, created by FreeSurfer's recon-all on your sMRI images, or the directory containing the pickled data. Must be given unless -t is used and the input pkl file already exists.")
    parser.add_argument('-n', '--neigh_count', help="Number of vertices to consider at max in the edge neighborhoods for Euclidean dist.", default="500")
    parser.add_argument('-r', '--neigh_radius', help="Radius for sphere for Euclidean dist, in spatial units of mesh (e.g., mm).", default="10")
    parser.add_argument('-l', '--load_max', help="Total number of samples to load. Set to 0 for all in the files discovered in the data_dir. Used in sequential mode only.", default="0")
    parser.add_argument('-p', '--load_per_file', help="Total number of samples to load per file. Set to 0 for all in the respective mesh file. Useful to sample data from more different subjects and still not exhaust your RAM.", default="50000")
    parser.add_argument('-f', '--load_files', help="Total number of files to load. Set to 0 for all in the data_dir. Used in parallel mode only (see -s).", default="96")
    parser.add_argument("-s", "--sequential", help="Load data sequentially (as opposed to in parallel, the default). Not recommended. See also '-c'.", action="store_true")
    parser.add_argument("-c", "--cores", help="Number of cores to use when loading data in parallel. Defaults to all. (Model fitting always uses all cores.)", default=None)
    parser.add_argument("-t", "--pickle_tag", help="Optional, a tag (arbitrary string that will become a filename part) if you want to use pickling (saving/restoring) for datasets. If given the tag will be used to construct 1) the filename from/to which to unpickle/pickle the pre-processed dataset as 'ml<dataset_tag>_dataset.pkl', and 2) of the JSON metadata file for the dataset as 'ml<dataset_tag>_dataset.json'. If the model file does not exist, it will be created during the first run (with the respective JSON file), and used in subsequent runs with the same '--pickle-tag'. Can save a lot of time during model tuning if the dataset is final. Example: '_lgbmv1'.", default="")
    parser.add_argument("-w", "--write_dir", help="Optional writeable directory in which to save and from which to load pickled models and datasets, instead of in the data_dir. Useful if the latter is needed for the source data but is read-only. Ignored unless '-t' is also specified.")
    args = parser.parse_args()

    # Data settings not exposed on cmd line. Change here if needed.
    surface = 'pial'  # The mesh to use.
    descriptor = 'pial_lgi'  # The label descriptor, what you want to predict on the mesh.
    random_state = 42
    load_per_file_force_exactly = True # Whether to load exactly the requested number of entries per file, even if the file contains more (and more where thus read when reading it).

    # Preproc settings which are not exposed on the command line. (They are not exposed because changing them is most likely not need or a bad idea).
    cortex_label = False  # Whether to load FreeSurfer 'cortex.label' files and filter verts by them. Not implemented yet.
    add_desc_vertex_index = True  # whether to add vertex index as desriptor column to observation
    add_desc_neigh_size = True  # whether to add vertex neighborhood size (before pruning) as desriptor column to observation
    filter_smaller_neighborhoods = False  # Whether to filter (remove) neighborhoods smaller than 'args.neigh_count' (True), or fill the missing columns with 'np.nan' values instead. Note that, if you set to False, you will have to deal with the NAN values in some way before using the data, as most ML models cannot cope with NAN values.
    add_desc_brain_bbox = True
    add_local_mesh_descriptors = True
    add_global_mesh_descriptors = True

    # Data post-processing options (stuff that happens after loading).
    # These should become part of the data pre-processing pipeline (and preproc_settings), but that requires us to load training and
    # eval/testing data completely separately, which is not done yet. (The current method is okay,
    # it's just not that beautiful, and we need to take during data post-processing not to cause any leakage).
    # We also need to make sure to apply the same settings during prediction.
    do_replace_nan = postproc_settings['do_replace_nan']
    replace_nan_with = postproc_settings['replace_nan_with']
    do_scale_descriptors = postproc_settings['do_scale_descriptors']
    scale_func = postproc_settings['scale_func']

    ### Construct data settings from command line and other data setting above.

    ## All settings relevant for pre-processing of a single mesh. These must also be used when pre-processing meshes that you want to predict pvd-descriptors for later.
    preproc_settings = { 'cortex_label': cortex_label,
                        'add_desc_vertex_index':add_desc_vertex_index,
                        'add_desc_neigh_size':add_desc_neigh_size,
                        'mesh_neighborhood_radius':int(args.neigh_radius),
                        'mesh_neighborhood_count':int(args.neigh_count),
                        'filter_smaller_neighborhoods': filter_smaller_neighborhoods,
                        'add_desc_brain_bbox': add_desc_brain_bbox,
                        'add_local_mesh_descriptors' : add_local_mesh_descriptors,
                        'add_global_mesh_descriptors': add_global_mesh_descriptors
                    }

    ## All settings relevant for deciding which meshes to load, how to load them, and what data to keep from them.
    data_settings_in = {'data_dir': args.data_dir,
                        'num_neighborhoods_to_load': None if int(args.load_max) == 0 else int(args.load_max),
                        'surface': surface,
                        'descriptor' : descriptor,
                        'verbose': args.verbose,
                        'sequential': args.sequential,
                        'num_samples_per_file': None if int(args.load_per_file) == 0 else int(args.load_per_file),
                        'num_cores': None if (args.cores is None or args.cores == "0") else int(args.cores),
                        'num_files_to_load':None if int(args.load_files) == 0 else int(args.load_files),
                        'exactly': load_per_file_force_exactly
                        }


    ### Other settings, not related to data loading. Adapt here if needed.
    do_pickle_data = len(args.pickle_tag) > 0

    # Some common thing to identify a certain dataset. Freeform. Set to empty string if you do not need this.
    # Allows switching between pickled datasets quickly.
    dataset_tag = args.pickle_tag if args.pickle_tag is not None else ""
    model_tag = dataset_tag

    write_dir = args.data_dir if args.write_dir is None else args.write_dir

    dataset_pickle_file = os.path.join(write_dir, f"ml{dataset_tag}_dataset.pkl")  # Only relevant if do_pickle_data is True
    dataset_settings_file = os.path.join(write_dir, f"ml{dataset_tag}_dataset.json") # Only relevant if do_pickle_data is True
    training_history_image_filename = os.path.join(write_dir, f"ml{dataset_tag}_training.png")  # Image to save training history.

    do_persist_trained_model = True
    model_save_file = os.path.join(write_dir, f"ml{model_tag}_model.pkl")
    model_settings_file = os.path.join(write_dir, f"ml{model_tag}_model.json")
    num_cores_fit = None

    # Model settings
    lightgbm_num_estimators = 144 * 3  # The number of estimators (trees) to use during final model fitting.
    do_hyperparam_opt = False  # Dramatically increases computational time (depends on hyperparm opt settings, but 60 times to 200 times is typical). Do this ONCE on a medium sized dataset, copy the obtained params and hard-code them in the source code of the fit function to re-use (and set this to FALSE then).
    hyper_tune_num_iter = 20   # Number of search iterations for hyperparam tuning. Only used when do_hyperparam_opt=True.
    hyper_tune_inner_cv_k = 3  # The k for k-fold cross-validation during hyperparam tuning. Only used when do_hyperparam_opt=True.
    hpt_num_estimators = 100   # The number of estimators (trees) to use during hyperparam tuning, equivalent to `lightgbm_num_estimators`. Only used when do_hyperparam_opt=True

    # Obtained from hyperparam optimization run, hard-coded here.
    opt_fit_settings = {'colsample_bytree': 0.8532168461905915, 'min_child_samples': 489, 'min_child_weight': 10.0,
                                                        'num_leaves': 47, 'reg_alpha': 2, 'reg_lambda': 20, 'subsample': 0.22505063396444688}

    ####################################### End of settings. #########################################

    will_load_dataset_from_pickle_file = do_pickle_data and os.path.isfile(dataset_pickle_file)

    print("---Train and evaluate an lGI prediction model---")


    if data_settings_in['verbose']:
        print("Verbosity turned on.")
        if (not will_load_dataset_from_pickle_file) and args.write_dir is not None:
            print(f"Parameter 'datadir' has no effect in current settings and is implicitely assumed to be the current working directory.")
        if do_hyperparam_opt:
            print(f"Will perform hyperparameter tuning with {hyper_tune_num_iter} iterations and {hyper_tune_inner_cv_k}-fold cross-validation ({hyper_tune_num_iter * hyper_tune_inner_cv_k} runs). ")
        else:
            print(f"Will NOT perform hyperparameter tuning, using hard-coded opt_fit_settings: '{opt_fit_settings}'.")
        print(f"Will use {lightgbm_num_estimators} estimators to fit (final) model with {num_cores_fit} cores.")
        if do_pickle_data:
            print(f"Using dataset_tag '{dataset_tag}' and model_tag '{model_tag}' for filenames when loading/saving data and model.")

    num_cores_tag = "all" if data_settings_in['num_cores'] is None or data_settings_in['num_cores'] == 0 else data_settings_in['num_cores']
    seq_par_tag = " sequentially " if data_settings_in['sequential'] else f" in parallel using {num_cores_tag} cores"

    if not will_load_dataset_from_pickle_file:
        if data_settings_in['verbose']:
            if add_desc_brain_bbox:
                print(f"Will add brain bounding box coords as extra descriptor columns.")
            else:
                print(f"Will NOT add brain bounding box coords as extra descriptor columns.")

            if data_settings_in['sequential']:
                print(f"Loading datafiles{seq_par_tag}.")
                print(f"Using data directory '{data_settings_in['data_dir']}', observations to load total limit is set to: {data_settings_in['num_neighborhoods_to_load']}.")
            else:
                print("Loading datafiles in parallel.")
                print(f"Using data directory '{data_settings_in['data_dir']}', number of files to load limit is set to: {data_settings_in['num_files_to_load']}.")

            print(f"Using neighborhood radius {preproc_settings['mesh_neighborhood_radius']} and keeping {preproc_settings['mesh_neighborhood_count']} vertices per neighborhood.")

            print("Descriptor settings:")
            if add_desc_vertex_index:
                print(f" - Adding vertex index in mesh as additional descriptor (column) to computed observations (neighborhoods).")
            else:
                print(f" - Not adding vertex index in mesh as additional descriptor (column) to computed observations (neighborhoods).")
            if add_desc_neigh_size:
                print(f" - Adding neighborhood size before pruning as additional descriptor (column) to computed observations (neighborhoods).")
            else:
                print(f" - Not adding neighborhood size before pruning as additional descriptor (column) to computed observations (neighborhoods).")

            mem_avail_mb = int(psutil.virtual_memory().available / 1024. / 1024.)
            print(f"RAM available is about {mem_avail_mb} MB.")
            can_estimate = False
            ds_estimated_num_neighborhoods = None
            ds_estimated_num_values_per_neighborhood = 6 * preproc_settings['mesh_neighborhood_count'] + 1  # minor TODO: The +1 is not true (depends on settings above), but this is minor in comparison to 6 * data_settings_in['mesh_neighborhood_count'] anyways.
            if data_settings_in['num_neighborhoods_to_load'] is not None and data_settings_in['sequential']:
                # Estimate total dataset size in RAM early to prevent crashing later, if possible.
                ds_estimated_num_neighborhoods = data_settings_in['num_neighborhoods_to_load']
                can_estimate = True
            if data_settings_in['num_samples_per_file'] is not None and data_settings_in['num_files_to_load'] is not None and not data_settings_in['sequential']:
                ds_estimated_num_neighborhoods = data_settings_in['num_samples_per_file'] * data_settings_in['num_files_to_load']
                can_estimate = True
            if can_estimate:
                # try to allocate, will err if too little RAM.
                ds_dummy = np.empty((ds_estimated_num_neighborhoods, ds_estimated_num_values_per_neighborhood), dtype=np.float32)
                ds_estimated_full_data_size_bytes = getsizeof(ds_dummy)
                ds_dummy = None
                del ds_dummy
                gc.collect()
                ds_estimated_full_data_size_MB = ds_estimated_full_data_size_bytes / 1024. / 1024.
                print(f"Estimated dataset size in RAM will be {int(ds_estimated_full_data_size_MB)} MB.")
                if ds_estimated_full_data_size_MB * 2.0 >= mem_avail_mb:
                    print(f"WARNING: Dataset size in RAM is more than half the available memory!") # A simple copy operation will lead to trouble!


    dataset, _, data_settings = get_dataset_pickle(data_settings_in, preproc_settings, do_pickle_data, dataset_pickle_file, dataset_settings_file)

    if data_settings_in['verbose']:
        print(f"Obtained dataset of {int(getsizeof(dataset) / 1024. / 1024.)} MB, containing {dataset.shape[0]} observations, and {dataset.shape[1]} columns ({dataset.shape[1]-1} features + 1 label). {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.")

    dataset_postproc_start = time.time()

    # Shuffle the entire dataset, to prevent the model from training only on (consecutive) vertices from some of the meshes in the set of input files.
    if data_settings_in['verbose']:
        print(f"Shuffling the rows (row order) of the dataframe.")
    #dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    from sklearn.utils import shuffle # We use sklearn.utils.shuffle over pandas.DataFrame.sample, as that is buggy in my pandas version and allocates lots of memory (more than 2x size of MB in RAM), crashing this script for large datasets.
    dataset = shuffle(dataset, random_state=random_state)
    dataset.reset_index(inplace=True, drop=True)


    ### NAN handling. Only needed if 'filter_smaller_neighborhoods' is False.
    # WARNING: If doing non-trivial stuff, perform this separately on the train, test and evaluation data sets to prevent leakage!
    if do_replace_nan:
        row_indices_with_nan_values = pd.isnull(dataset).any(1).to_numpy().nonzero()[0]
        if row_indices_with_nan_values.size > 0:
            if data_settings_in['verbose']:
                print(f"NOTICE: Dataset contains {row_indices_with_nan_values.size} rows (observations) with NAN values (of {dataset.shape[0]} observations total).")
            dataset = dataset.fillna(replace_nan_with, inplace=False) # TODO: replace with something better? Like col mean? But if you do that, do NOT do it here, on the entire dataset! In that case, it has to be done separately on the test, train and eval datasets.
            if data_settings_in['verbose']:
                print(f"Filling NAN values in {row_indices_with_nan_values.size} columns with value '{replace_nan_with}'.")
            row_indices_with_nan_values = pd.isnull(dataset).any(1).to_numpy().nonzero()[0]
            if data_settings_in['verbose']:
                print(f"Dataset contains {row_indices_with_nan_values.size} rows (observations) with NAN values (of {dataset.shape[0]} observations total) after filling. {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.")
        else:
            if data_settings_in['verbose']:
                print(f"Dataset contains no NAN values. {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.")
        del row_indices_with_nan_values
    else:
        if data_settings_in['verbose']:
            print(f"Not trying to replace NAN values (if any).")


    nc = len(dataset.columns)
    feature_names = np.array(dataset.columns[:-1]) # We require that the label is in the last column of the dataset.
    label_name = dataset.columns[-1]
    print(f"Separating observations into {len(feature_names)} features and target column '{label_name}'...")

    dataset = dataset.to_numpy()

    y = dataset[:, (nc-1)]
    dataset = dataset[:, 0:(nc-1)]
    X = dataset
    del dataset
    gc.collect()

    print(f"Splitting data into train and test sets... ({int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    X = None # Free RAM.
    y = None
    del X, y
    gc.collect()

    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
    print(f"Created validation data set with shape {X_eval.shape}.")

    print(f"Created training data set with shape {X_train.shape} and testing data set with shape {X_test.shape}. {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.")
    print(f"The label arrays have shape {y_train.shape} for the training data and  {y_test.shape} for the testing data.")


    if data_settings_in['verbose']:
        dataset_postproc_end = time.time()
        dataset_postproc_execution_time = dataset_postproc_end - dataset_postproc_start
        print(f"=== Post-processing dataset done (shuffle, NAN-fill, train/test/validation-split), it took: {timedelta(seconds=dataset_postproc_execution_time)} ===")

    if do_scale:
        print(f"Scaling... (Started at {time.ctime()}, {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.)")

        #sc = StandardScaler()
        #X_train = sc.fit_transform(X_train)
        #X_test = sc.transform(X_test)
        #X_eval = sc.transform(X_eval)

        X_train = scale_func(X_train)
        X_test = scale_func(X_test)
        X_eval = scale_func(X_eval)


    print(f"Fitting with LightGBM Regressor with {lightgbm_num_estimators} estimators on {num_cores_fit} cores. (Started at {time.ctime()}.)")
    if do_hyperparam_opt:
        model, model_info = hyperparameter_optimization_lightgbm(X_train, y_train, X_eval, y_eval, num_iterations=hyper_tune_num_iter, inner_cv_k=hyper_tune_inner_cv_k, hpt_num_estimators=hpt_num_estimators, num_cores=num_cores_fit, random_state=random_state, eval_metric="neg_mean_absolute_error", verbose_lightgbm=1, verbose_random_search=2)
    else:
        model_settings_lightgbm = {'n_estimators':lightgbm_num_estimators, 'random_state':random_state, 'n_jobs':num_cores_fit}
        model, model_info = fit_regression_model_lightgbm(X_train, y_train, X_eval, y_eval, model_settings=model_settings_lightgbm, opt_fit_settings=opt_fit_settings)

    ax = lightgbm.plot_metric(model)
    try:
        plt.savefig(training_history_image_filename)
    except Exception as ex:
        print(f"Could not save training history plot to file '{training_history_image_filename}': {str(ex)}")

    model_info = eval_model_train_test_split(model, model_info, X_test, y_test, X_train, y_train, X_eval=X_eval, y_eval=y_eval)

    ## Assess feature importance (if possible)
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    model_info = report_feature_importances(importances, feature_names, model_info, num_to_report=30)


    model_and_data_info = { 'data_settings' : data_settings, 'model_info' : model_info }
    if do_persist_trained_model:
        save_model(model, model_and_data_info, model_save_file, model_settings_file)

        ## Some time later, load the model.
        #model, model_and_data_info = load_model(model_save_file, model_settings_file)
        #result = model.score(X_test, Y_test)

if __name__ == "__main__":
    train_lgi()
