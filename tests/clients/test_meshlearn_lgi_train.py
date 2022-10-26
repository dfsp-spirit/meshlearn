# Tests for the meshlearn_lgi_train script.
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
    ret = script_runner.run('meshlearn_lgi_train', '--help')
    assert ret.success
    assert 'usage' in ret.stdout
    assert ret.stderr == ''

# We set the env var MESHLEARN_TESTS_ON_GITHUB in our Github workflow file, at <repo>/.github/workflows/*.
@pytest.mark.slow  # Use `pytest -v -m "not slow"` to exlude all tests marked as 'slow'.
@pytest.mark.skipif(os.getenv("MESHLEARN_TESTS_ON_GITHUB", "false") == "true", reason="Not enough memory on Github, see issue #13.")
def test_train_sequential(script_runner):
    """
    Train model. Note that the amount of training data used here is very limited and totally unsuitable to train a well-performing model.
    This is just to make the unit test run quickly.
    """
    data_dir = os.path.join(TEST_DATA_DIR, 'abide_lgi')
    with tempfile.TemporaryDirectory() as tmpdir_name:
        ret = script_runner.run('meshlearn_lgi_train', '-v', '-n', '50', '-r', '10', '-l', '100000', '-w', tmpdir_name, data_dir)
    assert ret.success
