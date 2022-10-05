# -*- coding: utf-8 -*-

import os

import pandas as pd
from meshlearn.data.curvature import Curvature
from meshlearn.data.mem_opt import reduce_mem_usage

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default:
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)


def test_mem_opt():
    mesh_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial')
    df = Curvature(mesh_file).compute(["gaussian_curvature", "mean_curvature"])
    assert isinstance(df, pd.DataFrame)

    df_opt = reduce_mem_usage(df)
    assert df.shape == df_opt.shape
    assert (df.columns == df_opt.columns).all()


