# -*- coding: utf-8 -*-

import pytest
import trimesh as tm
import os
import numpy as np
import pandas as pd
from meshlearn.data.neighborhood import _get_mesh_neighborhood_feature_count, neighborhoods_euclid_around_points
from meshlearn.data.training_data import TrainingData
from scipy.spatial import KDTree
from meshlearn.model.persistance import load_model, save_model
import tempfile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
# Respect the environment variable MESHLEARN_TEST_DATA_DIR if it is set. If not, fall back to default:
TEST_DATA_DIR = os.getenv('MESHLEARN_TEST_DATA_DIR', TEST_DATA_DIR)

@pytest.fixture
def model_files():
    model_pkl_file = os.path.join(TEST_DATA_DIR, 'models', 'lgbm_lgi', 'ml_model.pkl')
    metadata_json_file = os.path.join(TEST_DATA_DIR, 'models', 'lgbm_lgi', 'ml_model.json')  # Metadata file is not needed for predictions, return None if you do not have it.
    return model_pkl_file, metadata_json_file

def test_load(model_files):
    model_pkl_file, metadata_json_file = model_files
    model, model_and_data_info = load_model(model_pkl_file, metadata_json_file, verbose=True)
    assert isinstance(model_and_data_info, dict)


def test_save(model_files):
    model_pkl_file, metadata_json_file = model_files
    model, model_and_data_info = load_model(model_pkl_file, metadata_json_file, verbose=False)
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_save_file = os.path.join(tmp_dir, "model.pkl")
        model_settings_file = os.path.join(tmp_dir, "model.json")
        save_model(model, model_and_data_info, model_save_file, model_settings_file, verbose=True)
        assert os.path.isfile(model_settings_file)
        assert os.path.isfile(model_save_file)

