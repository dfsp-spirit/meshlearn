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
    assert isinstance(lgi_predicted, np.ndarray)

    num_mesh_vertices = 149244
    assert lgi_predicted.size == num_mesh_vertices
    assert np.min(lgi_predicted) >= 0.0
    assert np.max(lgi_predicted) <= 6.0
    lgi_known = fsio.read_morph_data(descriptor_file)
    assert np.corrcoef(lgi_predicted, lgi_known)[0,1] > 0.9  # Require high correlation.

def test_predict_with_list(test_file_pair, model_files):
    mesh_file, descriptor_file = test_file_pair
    model_pkl_file, metadata_json_file = model_files

    Mp = MeshPredictLgi(model_pkl_file, metadata_json_file)
    lgi_predicted = Mp.predict([mesh_file, mesh_file])  # Our list is pretty lame.
    assert isinstance(lgi_predicted, list)
    assert len(lgi_predicted) == 2
    num_mesh_vertices = 149244

    lgi_known = fsio.read_morph_data(descriptor_file)
    for pred_values in lgi_predicted:
        assert isinstance(pred_values, np.ndarray)
        assert pred_values.size == num_mesh_vertices
        assert np.min(pred_values) >= 0.0
        assert np.max(pred_values) <= 6.0
        assert np.corrcoef(pred_values, lgi_known)[0,1] > 0.9  # Require high correlation.

#@pytest.mark.skip(reason="I'm in a hurry.")
def test_predict_for_recon_dir(model_files):
    model_pkl_file, metadata_json_file = model_files
    recon_dir = os.path.join(TEST_DATA_DIR, 'abide_lgi')

    Mp = MeshPredictLgi(model_pkl_file, metadata_json_file)
    subjects_list=["Leuven_2_0050743"]
    pvd_files_written, infiles_okay, infiles_missing, infiles_with_errors = Mp.predict_for_recon_dir(recon_dir, subjects_list=subjects_list)

    assert len(infiles_okay) + len(infiles_missing) + len(infiles_with_errors) == len(subjects_list) * 2  # For the 2 hemis.
    assert len(infiles_okay) == len(subjects_list) * 2 # For the 2 hemis.
    for f in pvd_files_written:
        os.remove(f)



