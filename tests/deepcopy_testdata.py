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
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument("-m", "--file-list", help="A text file listing one file to copy per line.")
    file_group.add_argument('-f', '--file', nargs=1, help="The file to copy, relative to <source_dir>/<subject>/.", default="surf/lh.pial")
    parser.add_argument('-l', '--subjects-file', nargs=1, help="The subjects file to use. Text file with one subject per line. If left at default, <source_dir>/subjects.txt is used.", default="_")
    parser.add_argument('-d', '--descriptor', nargs=1, help="The descriptor to copy.", default="pial_lgi")
    args = parser.parse_args()


    print("---Copy training/test data from a FreeSurfer recon-all output directory---")
    if args.verbose:
        print("Verbosity turned on.")

    if not os.path.exists(args.source_dir):
        raise ValueError("The source directory '{source_directory}' does not exist or cannot be accessed".format(source_directory=args.source_dir))

    if not os.path.exists(args.target_dir):
        raise ValueError("The target directory '{target_directory}' does not exist or cannot be accessed".format(target_directory=args.target_dir))

    subjects_file = args.subjects_file
    if subjects_file == "_":
        subjects_file = os.path.join(args.source_directory, 'subjects.txt')
        if args.verbose:
            print("Assuming subjects file '{subjects_file}'".format(subjects_file=subjects_file))

    if not os.path.isfile(subjects_file):
        raise ValueError("The file '{subjects_file}' does not exist or cannot be read".format(subjects_file=subjects_file))

    subjects_list = nit.read_subjects_file(subjects_file)

    files_per_subject = []
    if args.file:
        files_per_subject = [args.file]
    else:
        files_per_subject = nit.read_subjects_file(args.file_list)    # It is not a subjects file, but the format is the same.

    for subject in subjects_list:
        for sfile_rel in files_per_subject:
            sfile = os.path.join(args.source_directory, subject, sfile_rel)
            if not os.path.isfile(sfile):
                print("WARNING: Expected file '{sfile}' does not exist or cannot be read.".format(sfile=sfile))
            else:
                
        







if __name__ == "__main__":
    deepcopy_testdata_freesurfer()