# -*- coding: utf-8 -*-

import pytest
import pandas as pd
from meshlearn.data.curvature import Curvature
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default:
TEST_DATA_DIR = os.getenv('MESHLEARN_TEST_DATA_DIR', TEST_DATA_DIR)


@pytest.fixture
def test_file_pair():
    mesh_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial')
    descriptor_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial_lgi')
    return mesh_file, descriptor_file

def test_curvature_instance_can_be_initialized(test_file_pair):
    mesh_file, _ = test_file_pair
    num_verts = 149244
    curv = Curvature(mesh_file)

    assert isinstance(curv.pc, dict)
    assert len(curv.pc) == 4
    assert curv.k1.size == num_verts
    assert curv.k2.size == num_verts
    assert curv.k_major.size == num_verts
    assert curv.k_minor.size == num_verts

def test_prinicipal_curvatures_can_be_computed(test_file_pair):
    mesh_file, _ = test_file_pair
    num_verts = 149244
    curv = Curvature(mesh_file)

    desc = curv.compute_all()
    assert isinstance(desc, pd.DataFrame)
    assert desc.columns.size == 20 # The 4 basic ones (k1, k2, k_major, k_minor) and the 16 other implemented ones.
    assert desc.shape == (num_verts, 20)

    desc = curv.compute(["gaussian_curvature", "mean_curvature"])
    assert isinstance(desc, pd.DataFrame)
    assert desc.columns.size == 6  # The 4 basic ones (k1, k2, k_major, k_minor) and the 2 requested ones.
    assert desc.shape == (num_verts, 6)

def test_save_curv(test_file_pair):
    mesh_file, _ = test_file_pair
    curv = Curvature(mesh_file)
    filenames = curv._save_curv(outdir=os.path.expanduser("~"))
    for f in filenames:
        assert os.path.isfile(f)
        os.remove(f)

def test_save_csv(test_file_pair):
    mesh_file, _ = test_file_pair
    curv = Curvature(mesh_file)
    out_file = "descriptors.csv"
    filenames = curv._save_csv(out_file)
    assert os.path.isfile(out_file)
    os.remove(out_file)





