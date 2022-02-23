#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import numpy as np
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit
import brainload.freesurferdata as fsd
import brainload.brainwrite as brw
import argparse
import glob

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# PYTHONPATH=./src/meshlearn python src/meshlearn/clients/meshlearn_lgi.py --verbose


def meshlearn_lgi():
    """
    Train and evaluate an lGI prediction model.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate an lGI prediction model.")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    parser.add_argument('-d', '--data-dir', help="The data directory. Use deepcopy_testdata.py script to create.", default="./tests/test_data/tim_only")
    parser.add_argument('-e', '--epochs', help="Number of training epochs.", default="20")
    args = parser.parse_args()

    num_epochs = int(args.epochs)
    data_dir = args.data_dir

    print("---Train and evaluate an lGI prediction model---")
    if args.verbose:
        print("Verbosity turned on.")
        print("Training for {num_epochs} epochs.".format(num_epochs=num_epochs))
        print("Using data directory '{data_dir}'.".format(data_dir=data_dir))

    if not os.path.isdir(data_dir):
        raise ValueError("The data directory '{data_dir}' does not exist or cannot be accessed".format(data_dir=data_dir))

    mesh_files = np.sort(glob.glob("{data_dir}/*.pial".format(data_dir=data_dir)))
    descriptor_files = np.sort(glob.glob("{data_dir}/*.pial_lgi".format(data_dir=data_dir)))
    if args.verbose:
        print("Found {num_mesh_files} mesh files: {mesh_files}".format(num_mesh_files=len(mesh_files), mesh_files=', '.join(mesh_files)))
        print("Found {num_descriptor_files} descriptor files: {descriptor_files}".format(num_descriptor_files=len(descriptor_files), descriptor_files=', '.join(descriptor_files)))

    file_pairs = dict(zip(mesh_files, descriptor_files))
    for mesh_filename, desc_filename in file_pairs.items():
        expected_desc_filename = "{mesh_filename}_lgi".format(mesh_filename=mesh_filename)
        if desc_filename != expected_desc_filename:
            raise ValueError("Mesh file '{mesh_filename}' should have matching descriptor file named '{expected_desc_filename}', but was matched to '{desc_filename}'.".format(mesh_filename=mesh_filename, expected_desc_filename=expected_desc_filename, desc_filename=desc_filename))

    if args.verbose:
        print("All mesh files seem to have the expected descriptor files associated with them.")

    




    sys.exit(0)


if __name__ == "__main__":
    meshlearn_lgi()
