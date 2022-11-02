# -*- coding: utf-8 -*-

import os
import pytest
import pandas as pd
from meshlearn.data.generator import neighborhood_generator_filepairs, neighborhood_generator_reconall_dir
from meshlearn.util.recon import get_valid_mesh_desc_file_pairs_reconall

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
# Respect the environment variable MESHLEARN_TEST_DATA_DIR if it is set. If not, fall back to default:
TEST_DATA_DIR = os.getenv('MESHLEARN_TEST_DATA_DIR', TEST_DATA_DIR)


@pytest.fixture
def preproc_settings():
    return {
            'cortex_label': False,
            'add_desc_vertex_index': True,
            'add_desc_neigh_size': True,
            'mesh_neighborhood_radius': 10,
            'mesh_neighborhood_count': 300,
            'filter_smaller_neighborhoods': False,
            'add_desc_brain_bbox': True,
            'add_local_mesh_descriptors' : True,
            'add_global_mesh_descriptors': True
            }

@pytest.fixture
def data_settings():
    abide_dir = os.path.join(TEST_DATA_DIR, "abide_lgi")
    return {
            'data_dir': abide_dir,
            'surface' : "pial",
            'descriptor' : 'pial_lgi',
            'num_samples_total': None,
            'num_samples_per_file': None,
            'random_seed': None,
            'exactly': False,
            'reduce_mem': False  # Reducing mem takes a lot of time, is not needed here and we want fast predictions.
            }


def test_neighborhood_generator_filepairs(data_settings, preproc_settings):
    batch_size = 10000
    verbose=True

    mesh_files, desc_files, cortex_files, files_subject, files_hemi, miss_subjects = get_valid_mesh_desc_file_pairs_reconall(data_settings['data_dir'], surface=data_settings['surface'], descriptor=data_settings['descriptor'], cortex_label=preproc_settings.get('cortex_label', False), verbose=data_settings.get("verbose", True), subjects_file=data_settings.get('subjects_file', None), subjects_list=data_settings.get('subjects_list', None), hemis=data_settings.get('hemis', ["lh", "rh"]))
    input_filepair_list = list(zip(mesh_files, desc_files))
    gen = neighborhood_generator_filepairs(batch_size, input_filepair_list, preproc_settings=preproc_settings, verbose=verbose)
    for i in range(5):
        data = next(gen)
        assert isinstance(data, pd.DataFrame)


def test_neighborhood_generator_reconall_dir(data_settings, preproc_settings):
    batch_size = 10000
    verbose=True

    gen = neighborhood_generator_reconall_dir(batch_size, data_settings=data_settings, preproc_settings=preproc_settings, verbose=verbose)
    for i in range(5):
        data = next(gen)
        assert isinstance(data, pd.DataFrame)
