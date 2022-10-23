# -*- coding: utf-8 -*-

import pytest
import os
import numpy as np
from meshlearn.model.predict import MeshPredictLgi
import nibabel.freesurfer.io as fsio


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default:
TEST_DATA_DIR = os.getenv('MESHLEARN_TEST_DATA_DIR', TEST_DATA_DIR)


@pytest.fixture
def test_file_pair():
    mesh_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial')
    descriptor_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial_lgi')
    return mesh_file, descriptor_file

@pytest.fixture
def model_files():
    model_pkl_file = os.path.join(TEST_DATA_DIR, 'models', 'lgbm_lgi', 'ml_model.pkl')
    metadata_json_file = os.path.join(TEST_DATA_DIR, 'models', 'lgbm_lgi', 'ml_model.json')  # Metadata file is not needed for predictions, return None if you do not have it.
    return model_pkl_file, metadata_json_file

#@pytest.mark.skip(reason="I'm in a hurry.")
def test_predict(test_file_pair, model_files):
    mesh_file, descriptor_file = test_file_pair
    model_pkl_file, metadata_json_file = model_files
    Mp = MeshPredictLgi(model_pkl_file, metadata_json_file)
    lgi_predicted = Mp.predict(mesh_file)
    num_mesh_vertices = 149244
    assert lgi_predicted.size == num_mesh_vertices
    assert np.min(lgi_predicted) >= 0.0
    assert np.max(lgi_predicted) <= 6.0
    lgi_known = fsio.read_morph_data(descriptor_file)
    assert np.corrcoef(lgi_predicted, lgi_known)[0,1] > 0.9  # Require high correlation.


