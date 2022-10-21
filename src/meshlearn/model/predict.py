# -*- coding: utf-8 -*-

"""
Model prediction functions.

This file is part of meshlearn, see https://github.com/dfsp-spirit/meshlearn for details.
"""

from abc import ABC, abstractmethod
import numpy as np
from meshlearn.model.persistance import load_model
from meshlearn.data.training_data import compute_dataset_for_mesh
import time
from datetime import timedelta

class MeshPredict():
    """
    Predict per-vertex descriptor values for a mesh.

    Parameters
    ----------
    mesh_file  : str, path to input mesh file.
    descriptor : str, the desciptor name. Currently only 'pial_lgi' is supported.
    """

    model = None
    model_settings = None
    verbose = True

    @abstractmethod
    def predict(self, mesh_file):
        pass


class MeshPredictLgi(MeshPredict):
    """
    Predict lGI for a mesh file.
    """
    def __init__(self, model_file, model_settings_file, verbose=True):
        self.verbose = verbose
        if isinstance(model_file, str) and isinstance(model_settings_file, str):
            self.model, self.model_settings = load_model(model_file, model_settings_file, verbose=self.verbose)
        elif isinstance(model_file, str) and (model_settings_file is None or isinstance(model_settings_file, dict)):
                self.model, _ = load_model(model_file, model_settings_file=None, verbose=self.verbose)
                self.model_settings = model_settings_file  # Assume it is a dict with the model info (or None).
        if hasattr(model_file, 'predict'):
                self.model = model_file  # Assume it is a model that has an sklearn-like API. It must support `x.precict()`.
        if isinstance(model_settings_file, dict):
            self.model_settings = model_settings_file  # Assume it is a dict with the model settings.

        if not hasattr(self.model, 'predict'):
            raise ValueError("Model obtained based on parameter 'model_file' has no 'model.predict' method. Pass a valid sklearn model or str representing a path to a pkl file containing one.")
        if not isinstance(self.model_settings, dict):
            raise ValueError("Model settings obtained based on parameter 'model_settings_file' is not a dict. Cannot perform pre-processing for the model if we do not have its settings.")
        if not 'data_settings' in self.model_settings:
            raise ValueError("Model settings obtained based on parameter 'model_settings_file' is a dict but is missing key 'data_settings' with data pre-processing settings. Cannot perform pre-processing of data for prediction if we do not have these settings.")

    def _extract_preproc_settings(self, legacy_model_settings):
        if 'preproc_settings' in legacy_model_settings:  # Not legacy.
            return legacy_model_settings['preproc_settings']
        else:
            preproc_settings = dict()
            keys_of_interest = ["add_desc_brain_bbox", "add_desc_neigh_size", "add_desc_vertex_index", "cortex_label", "filter_smaller_neighborhoods", "mesh_neighborhood_count" , "mesh_neighborhood_radius"]
            for k in keys_of_interest:
                if k in legacy_model_settings['data_settings']:
                    preproc_settings[k] = legacy_model_settings['data_settings'][k]
        return preproc_settings

    def predict(self, mesh_file):

        if self.verbose:
            preproc_start = time.time()

        preproc_settings = self._extract_preproc_settings(self.model_settings)
        dataset, _, _ = compute_dataset_for_mesh(mesh_file, preproc_settings)

        if self.verbose:
            preproc_end = time.time()
            preproc_execution_time = preproc_end - preproc_start
            preproc_execution_time_readable = timedelta(seconds=preproc_execution_time)
            print(f"Pre-processing mesh took {preproc_execution_time_readable}.")
            predict_start = time.time()

        res = self.model.predict(dataset)
        print(f"Predicted {res.size} lgi values in range {np.min(res)} to {np.max(res)}.")

        if self.verbose:
            predict_end = time.time()
            predict_execution_time = predict_end - predict_start
            predict_execution_time_readable = timedelta(seconds=predict_execution_time)
            print(f"Prediction of {dataset.shape[0]} values took {predict_execution_time_readable}.")
        return res




