# -*- coding: utf-8 -*-

"""
Model persistence functions.
"""

import pickle
import time
import json
from datetime import timedelta

def save_model(model, model_and_data_info, model_save_file, model_settings_file):
    """Save a model, and optionally metadata, to pickle and JSON files."""

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
        try:
            with open(model_settings_file, 'w') as fp:
                json.dump(model_and_data_info, fp, sort_keys=True, indent=4)
        except Exception as ex:
            print(f"NOTICE: Could not save model_and_data_info to file '{model_settings_file}': {str(ex)}.")

    pickle_model_end = time.time()
    pickle_model_save_time = pickle_model_end - pickle_model_start
    print(f"INFO: Saved trained model to pickle file '{model_save_file}', ready to load later. Saving model took {timedelta(seconds=pickle_model_save_time)}.")


def load_model(model_save_file, model_settings_file):
    """Load a pickled model and, if available, JSON metadata from files."""
    if not model_save_file.endswith('.pkl'):
            print(f"WARNING: Given model filename '{model_save_file}' does not have '.pkl' file extension.")
    model = pickle.load(open(model_save_file, 'rb'))
    model_and_data_info = None
    if model_settings_file is not None:
        if not model_settings_file.endswith('.json'):
            print(f"WARNING: Given model metadata JSON file filename '{model_settings_file}' does not have '.json' file extension.")
        try:
            with open(model_settings_file, 'r') as fp:
                model_and_data_info = json.load(fp)
                print(f"INFO: Loaded settings used to create dataset from file '{model_settings_file}'.")
        except Exception as ex:
            model_and_data_info = None
            print(f"NOTICE: Could not load settings used to create dataset from file '{model_settings_file}': {str(ex)}.")
    else:
        print(f"INFO: Not loading metadata for model, no filename given.")
    return model, model_and_data_info