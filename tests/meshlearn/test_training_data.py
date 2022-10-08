# -*- coding: utf-8 -*-

import pytest
import trimesh as tm
import os
import numpy as np
import pandas as pd
from meshlearn.data.training_data import TrainingData, get_valid_mesh_desc_lgi_file_pairs_flat_dir, get_valid_mesh_desc_file_pairs_reconall
from scipy.spatial import KDTree

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default:
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)


def test_get_valid_mesh_desc_lgi_file_pairs_flat_dir():
    valid_mesh_files, valid_desc_files = get_valid_mesh_desc_lgi_file_pairs_flat_dir(os.path.join(TEST_DATA_DIR, 'tim_only'), verbose=False)
    assert len(valid_mesh_files) == 2
    assert len(valid_desc_files) == 2


def test_get_valid_mesh_desc_file_pairs_reconall():
    valid_mesh_files, valid_desc_files, valid_labl_files, valid_files_subject, valid_files_hemi, subjects_missing_some_file = get_valid_mesh_desc_file_pairs_reconall(os.path.join(TEST_DATA_DIR, 'abide_lgi'), verbose=False)
    assert len(valid_mesh_files) == 4 * 2
    assert len(valid_desc_files) == 4 * 2
    assert len(valid_labl_files) == 0
    assert len(valid_files_subject) == len(valid_mesh_files)
    assert len(valid_files_hemi) == len(valid_mesh_files)
    assert len(subjects_missing_some_file) == 0