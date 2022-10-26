# -*- coding: utf-8 -*-

import os

import pandas as pd
from meshlearn.data.curvature import Curvature
from meshlearn.data.mem_opt import reduce_mem_usage
from sys import getsizeof
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
# Respect the environment variable MESHLEARN_TEST_DATA_DIR if it is set. If not, fall back to default:
TEST_DATA_DIR = os.getenv('MESHLEARN_TEST_DATA_DIR', TEST_DATA_DIR)


def test_mem_opt_float16_range():
    mesh_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial')
    df = Curvature(mesh_file).compute(["gaussian_curvature", "mean_curvature"])
    assert isinstance(df, pd.DataFrame)

    df_opt = reduce_mem_usage(df, verbose=True)
    assert df.shape == df_opt.shape
    assert (df.columns == df_opt.columns).all()
    assert getsizeof(df_opt) <= getsizeof(df)

def test_mem_opt_float():
    for dtype in [np.float32, np.float64]:
        low = np.finfo(dtype).max - 100.0
        data = np.random.default_rng().uniform(low=low, high=low+5.0, size=10)
        df = pd.DataFrame({"data": data})
        df_opt = reduce_mem_usage(df)
        assert df.shape == df_opt.shape


def test_mem_opt_int():
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        low = np.iinfo(dtype).max - 50
        data = np.random.default_rng().integers(low=low, high=low+30, size=10)
        df = pd.DataFrame({"data": data})
        df_opt = reduce_mem_usage(df)
        assert df.shape == df_opt.shape



