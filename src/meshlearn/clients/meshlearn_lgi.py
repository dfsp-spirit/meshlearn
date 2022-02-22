#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit
import brainload.freesurferdata as fsd
import brainload.brainwrite as brw
import argparse

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# SURF=tests/test_data/subject1/surf/lh.white
# PYTHONPATH=./src/meshlearn python src/meshlearn/clients/meshlearn_lgi.py -t $SURF


def meshlearn_lgi():
    """
    Train and evaluate an lGI prediction model.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate an lGI prediction model.")
    num_vert_group = parser.add_mutually_exclusive_group(required=True)     # we need some way to determine the number of vertices to use for the overlay
    num_vert_group.add_argument("-n", "--num-verts", help="The number of vertices in the target surface, an integer. Hint: the number for the Freesurfer fsaverage subject is 163842.")
    num_vert_group.add_argument("-t", "--target-surface-file", help="A target surface file to determine the number of vertices from. E.g., '<your_study>/subject1/surf/lh.white'.")
    index_group = parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("-i", "--index", help="The index of the vertex to query. A single integer or several integers separated by commata (no spaces allowed).")
    index_group.add_argument("-f", "--index-file", help="A file containing a list of vertex indices, one integer per line. Can optionally contain colors per vertex, then add 3 more integers per line, separated by commata. Example line: '0,255,0,0'")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    parser.add_argument('-c', '--color', nargs=3, help="The color to use for the vertices as 3 RGB values between 0 and 255, e.g., '255 0 0' for red. Must be given unless an index file is used that contains the color values.", default=None)
    parser.add_argument('-b', '--background-color', nargs=3, help="The background color to use for all the vertices which are NOT listed on the command line or in the index file. 3 RGB values between 0 and 255. Defaults to '128 128 128', a gray.", default=[128, 128, 128])
    parser.add_argument('-e', '--extend-neighborhood', nargs=2, help="In addition to the given vertices, also color their neighbors in the given mesh file, up to the given graph distance in hops.")
    parser.add_argument('-o', '--output-file', help="Ouput file. The format is an RGB overlay that can be loaded into Freeview.", default="surface_RGB_map.txt")
    args = parser.parse_args()


    print("---Train and evaluate an lGI prediction model---")
    if args.verbose:
        print("Verbosity turned on.")

    sys.exit(0)


if __name__ == "__main__":
    meshlearn_lgi()
