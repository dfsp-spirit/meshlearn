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