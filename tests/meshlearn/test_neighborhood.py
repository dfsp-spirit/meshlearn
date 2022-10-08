# -*- coding: utf-8 -*-

import pytest
import trimesh as tm
import os
import numpy as np
import pandas as pd
from meshlearn.data.neighborhood import _get_mesh_neighborhood_feature_count, neighborhoods_euclid_around_points
from meshlearn.data.training_data import TrainingData
from scipy.spatial import KDTree

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default:
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)


def test_get_mesh_neighborhood_feature_count():
    assert _get_mesh_neighborhood_feature_count(100) == 100 * 6
    assert _get_mesh_neighborhood_feature_count(100, with_normals=False) == 100 * 3
    assert _get_mesh_neighborhood_feature_count(100, with_label=True) == 100 * 6 + 1
    assert _get_mesh_neighborhood_feature_count(100, extra_fields=["descriptor1", "descriptor2"]) == 100 * 6 + 2


@pytest.fixture
def test_file_pair():
    mesh_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial')
    descriptor_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial_lgi')
    return mesh_file, descriptor_file


@pytest.fixture
def mesh_and_pvd_data(test_file_pair):
    mesh_file, descriptor_file = test_file_pair
    return TrainingData.data_from_files(mesh_file, descriptor_file)


def test_neighborhoods_euclid_around_points(mesh_and_pvd_data):
    vert_coords, faces, pvd_data = mesh_and_pvd_data
    max_num_neighbors = 100

    query_vert_coords = vert_coords[0:1000, :]
    num_query_coords = query_vert_coords.shape[0]
    query_vert_indices = np.arange(num_query_coords)
    mesh = tm.Trimesh(vertices=vert_coords, faces=faces)

    neighborhoods, col_names, kept_vertex_indices_mesh = neighborhoods_euclid_around_points(query_vert_coords, query_vert_indices, KDTree(vert_coords), neighborhood_radius=10.0, mesh=mesh, pvd_data=pvd_data, max_num_neighbors=max_num_neighbors, verbose=False, add_desc_vertex_index=False, add_desc_neigh_size=False)
    assert isinstance(neighborhoods, np.ndarray)
    assert neighborhoods.shape[0] <= num_query_coords # Some may have been filtered.
    assert len(col_names) == max_num_neighbors * 6 + 1
    assert kept_vertex_indices_mesh.size == neighborhoods.shape[0]

