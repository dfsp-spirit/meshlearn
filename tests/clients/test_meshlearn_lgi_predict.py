# Tests for the meshlearn_lgi_predict script.
#
# These tests require the package `pytest-console-scripts` that provides the 'script_runner' fixture.

import os
import pytest
import tempfile
import shutil

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
# Respect the environment variable MESHLEARN_TEST_DATA_DIR if it is set. If not, fall back to default:
TEST_DATA_DIR = os.getenv('MESHLEARN_TEST_DATA_DIR', TEST_DATA_DIR)

def test_help(script_runner):
    """
    Just trigger help.
    """
    ret = script_runner.run('meshlearn_lgi_predict', '--help')
    assert ret.success
    assert 'usage' in ret.stdout
    assert ret.stderr == ''

# Skip tests that write files to disk on CI.
# We set the env var MESHLEARN_TESTS_ON_GITHUB in our Github workflow file, at <repo>/.github/workflows/*.
@pytest.mark.slow  # Use `pytest -v -m "not slow"` to exlude all tests marked as 'slow'.
@pytest.mark.skipif(os.getenv("MESHLEARN_TESTS_ON_GITHUB", "false") == "true", reason="Not enough memory on Github, see issue #13.")
def test_predict_file_implicit_modelmetadata(script_runner):
    """
    Here we do not use '-j' to explicitely specify the metadata JSON file. It will be constructed from the model filename.
    """
    model_file = os.path.join(TEST_DATA_DIR, 'models', 'lgbm_lgi', 'ml_model.pkl')
    mesh_file = os.path.join(TEST_DATA_DIR, 'abide_lgi', 'Caltech_0051477', 'surf', 'lh.pial')
    with tempfile.TemporaryDirectory() as tmpdir_name:
        ret = script_runner.run('meshlearn_lgi_predict', '-p', mesh_file, '-d', tmpdir_name, model_file)
    assert ret.success

# Skip tests that write files to disk on CI.
# We set the env var MESHLEARN_TESTS_ON_GITHUB in our Github workflow file, at <repo>/.github/workflows/*.
@pytest.mark.slow  # Use `pytest -v -m "not slow"` to exlude all tests marked as 'slow'.
@pytest.mark.skipif(os.getenv("MESHLEARN_TESTS_ON_GITHUB", "false") == "true", reason="Not enough memory on Github, see issue #13.")
def test_predict_file_explicit_modelmetadata(script_runner):
    """
    Here we use '-j' to explicitely specify the metadata JSON file. We also run in verbose mode.
    """
    model_file = os.path.join(TEST_DATA_DIR, 'models', 'lgbm_lgi', 'ml_model.pkl')
    metadata_file = os.path.join(TEST_DATA_DIR, 'models', 'lgbm_lgi', 'ml_model.json')
    mesh_file = os.path.join(TEST_DATA_DIR, 'abide_lgi', 'Caltech_0051477', 'surf', 'lh.pial')
    with tempfile.TemporaryDirectory() as tmpdir_name:
        ret = script_runner.run('meshlearn_lgi_predict', '-p', mesh_file, '-j', metadata_file, '-d', tmpdir_name, '-v', model_file)
    assert ret.success
