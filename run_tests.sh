#!/bin/sh

if [ ! -d "tests/meshlearn/" ]; then
    echo "ERROR: Directory 'tests/meshlearn/' not found. Please run this script from the meshlab repo root. Exiting."
    exit 1
fi

export PYTHONPATH=$(pwd)
cd tests/
pytest

