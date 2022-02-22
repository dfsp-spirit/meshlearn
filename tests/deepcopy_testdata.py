#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import argparse
import brainload.nitools as nit
import shutil

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
        raise ValueError("The source directory '{source_dir}' does not exist or cannot be accessed".format(source_dir=args.source_dir))

    if not os.path.exists(args.target_dir):
        raise ValueError("The target directory '{target_dir}' does not exist or cannot be accessed".format(target_dir=args.target_dir))

    subjects_file = args.subjects_file
    if subjects_file == "_":
        subjects_file = os.path.join(args.source_dir, 'subjects.txt')
        if args.verbose:
            print("Assuming subjects file '{subjects_file}'".format(subjects_file=subjects_file))

    if not os.path.isfile(subjects_file):
        raise ValueError("The file '{subjects_file}' does not exist or cannot be read".format(subjects_file=subjects_file))

    subjects_list = nit.read_subjects_file(subjects_file)

    files_per_subject = []
    if args.file:
        files_per_subject = [args.file]
    else:
        files_per_subject = nit.read_subjects_file(args.file_list)    # It is not really a subjects file, but the format is the same.

    num_files_failed = 0
    for subject in subjects_list:
        for sfile_rel in files_per_subject:
            source_file = os.path.join(args.source_dir, subject, sfile_rel)
            subject_rel_dir = os.path.dirname(os.path.join(subject, sfile_rel)) # required below for reconstruction/creation of target path.
            dest_file = os.path.join(args.target_dir, subject, sfile_rel)
            if not os.path.isfile(source_file):
                print("WARNING: Expected source file '{source_file}' does not exist or cannot be read, skipping.".format(source_file=source_file))
                num_files_failed = num_files_failed + 1
            else:
                dest_subdir = os.path.join(args.target_dir, subject_rel_dir)
                if not os.path.exists(dest_subdir):
                    os.makedirs(dest_subdir, exist_ok=True)
                try:
                    shutil.copyfile(source_file, dest_file)
                except:
                    print("WARNING: Failed to copy source file '{source_file}' to destination '{dest_file}'.".format(source_file=source_file, dest_file=dest_file))
                    num_files_failed = num_files_failed + 1
    num_total = len(files_per_subject)*len(subjects_list)
    print("Done. {num_failed} of {num_total} files failed ({num_ok} okay).".format(num_failed=num_files_failed, num_total=num_total, num_ok=num_total-num_files_failed))


if __name__ == "__main__":
    deepcopy_testdata_freesurfer()