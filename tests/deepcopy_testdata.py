#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import argparse
import brainload.nitools as nit

def deepcopy_testdata_freesurfer():
    """
    Copy training/test data from a FreeSurfer recon-all output directory.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Copy training/test data from a FreeSurfer recon-all output directory.")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    parser.add_argument('-s', '--source-dir', nargs=1, help="The recon-all source subjects directory to copy the data from.", default=None)
    parser.add_argument('-t', '--target-dir', nargs=1, help="The target directory into which to copy the data.", default=None)
    parser.add_argument('-f', '--surface', nargs=1, help="The surface to copy.", default="pial")
    parser.add_argument('-l', '--subjects-file', nargs=1, help="The subjects file to use. Text file with one subject per line. If left at default, <source_dir>/subjects.txt is used.", default="_")
    parser.add_argument('-d', '--descriptor', nargs=1, help="The descriptor to copy.", default="pial_lgi")    
    args = parser.parse_args()


    print("---Copy training/test data from a FreeSurfer recon-all output directory---")
    if args.verbose:
        print("Verbosity turned on.")

    if not os.path.exists(args.source_dir):
        raise ValueError('The source directory {source_directory} does not exist or cannot be accessed'.format(source_directory=args.source_dir))

    if not os.path.exists(args.target_dir):
        raise ValueError('The target directory {target_directory} does not exist or cannot be accessed'.format(target_directory=args.target_dir))

    subjects_file = args.subjects_file
    if subjects_file == "_":
        subjects_file = os.path.join(args.source_directory, 'subjects.txt')

    if not os.path.isfile(subjects_file):
        raise ValueError('The file {subjects_file} does not exist or cannot be read'.format(subjects_file=subjects_file))

    subjects_list = nit.read_subjects_file(subjects_file)






if __name__ == "__main__":
    deepcopy_testdata_freesurfer()