# -*- coding: utf-8 -*-

"""
Model persistence functions.
"""

import pickle
import time
import json
import os
import numpy as np
from datetime import timedelta

def save_model(model, model_and_data_info, model_save_file, model_settings_file, verbose=True):
    """
    Save a model, and optionally metadata, to pickle and JSON files.

    Parameters
    ----------
    model               : the model to save, some sklearn or lightgbm model typically.
    model_and_data_info : dict or None, dict must be serializable to JSON. Can contain arbitrary model metadata to save in a separate JSON file.
    model_save_file     : str, the filename where to save the model (Pickle file), must end with '.pkl' or a warning will be printed. If it contains a path, that path must exist.
    model_settings_file : str or None, the filename where to save the model metadata (JSON), must end with '.json' or a warning will be printed. If it contains a path, that path must exist. Ignored if `model_and_data_info` is `None`.
    verbose             : bool, whether to print status info to stdout. Does not affect warnings, which will be printed even if `verbose=False`.

    Returns
    -------
    None, called for side effect of writing to disk.
    """

    # save the model to disk
    pickle_model_start = time.time()
    if not model_save_file.endswith('.pkl'):
        print(f"WARNING: Given model filename '{model_save_file}' does not have '.pkl' file extension.")
    pickle.dump(model, open(model_save_file, 'wb'))
    # Save the model settings as a JSON file.

    if model_and_data_info is not None:
        if not isinstance(model_and_data_info, dict):
            raise ValueError(f"Parameter 'model_and_data_info' must be a dict or None.")
        if not model_settings_file.endswith('.json'):
            print(f"WARNING: Given model metadata JSON file filename '{model_settings_file}' does not have '.json' file extension.")

        def check_keys(md, current_name="<root>", try_to_fix=True):
            for k in md:
                if isinstance(md[k], dict):
                    check_keys(md[k], current_name=f"{current_name}/{k}", try_to_fix=try_to_fix)
                if isinstance(md[k], (np.ndarray, np.number)):
                    print(f"[save_model]: WARNING 'model_and_data_info' entry '{k}' ({current_name}) is of type '{type(md[k])}' that cannot be serialized.")
                    if isinstance(md[k], np.number):
                        if try_to_fix:
                            try:
                                md[k] = md[k].item()
                            except Exception as ex:
                                print(f"NOTICE: Could not auto-fix entry '{k}' ({current_name}) of type '{type(md[k])}': {str(ex)}.")

        check_keys(model_and_data_info)

        try:
            with open(model_settings_file, 'w') as fp:
                json.dump(model_and_data_info, fp, sort_keys=True, indent=4)
        except Exception as ex:
            print(f"NOTICE: Could not save model_and_data_info to file '{model_settings_file}': {str(ex)}.")

    pickle_model_end = time.time()
    pickle_model_save_time = pickle_model_end - pickle_model_start
    pickle_file_size_mb = int(os.path.getsize(model_save_file) / 1024. / 1024.)
    if verbose:
        print(f"INFO: Saved trained model to pickle file '{model_save_file}' ({pickle_file_size_mb} MB), ready to load later. Saving model took {timedelta(seconds=pickle_model_save_time)}.")


def load_model(model_file, model_settings_file, verbose=True):
    """
    Load a pickled model and, if available, JSON metadata from files.

    Parameters
    ----------
    model_fil     e     : str, the filename from which to load the model (Pickle file), must end with '.pkl' or a warning will be printed.
    model_settings_file : str or None, the filename from which to load the model metadata (JSON), must end with '.json' or a warning will be printed. If it contains a path, that path must exist. Set to `None` if you do not have a metadata file or do not want to load it.
    verbose             : bool, whether to print status info to stdout. Does not affect warnings, which will be printed even if `verbose=False`.

    Returns
    -------
    model               : the loaded model, some sklearn or lightgbm model typically.
    model_and_data_info : dict or None. Contains the model metadata loaded from the JSON file.
    """
    if not model_file.endswith('.pkl'):
        print(f"WARNING: Given model filename '{model_file}' does not have '.pkl' file extension.")
    pickle_file_size_mb = int(os.path.getsize(model_file) / 1024. / 1024.)
    if verbose:
        print(f"Loading model from {pickle_file_size_mb} MB file '{model_file}'.")
    model = pickle.load(open(model_file, 'rb'))
    model_and_data_info = None
    if model_settings_file is not None:
        if not model_settings_file.endswith('.json'):
            print(f"WARNING: Given model metadata JSON file filename '{model_settings_file}' does not have '.json' file extension.")
        try:
            with open(model_settings_file, 'r') as fp:
                model_and_data_info = json.load(fp)
                if verbose:
                    print(f"INFO: Loaded settings used to create dataset from file '{model_settings_file}'.")
        except Exception as ex:
            model_and_data_info = None
            print(f"NOTICE: Could not load settings used to create dataset from file '{model_settings_file}': {str(ex)}.")
    else:
        if verbose:
            print(f"INFO: Not loading metadata for model, no filename given.")
    return model, model_and_data_info
