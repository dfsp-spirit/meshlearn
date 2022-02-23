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
    parser.add_argument('-s', '--source-dir', help="The recon-all source subjects directory to copy the data from.", required=True)
    parser.add_argument('-t', '--target-dir', help="The target directory into which to copy the data.", required=True)
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument("-m", "--file-list", help="A text file listing one file to copy per line.")
    file_group.add_argument('-f', '--file', help="The file to copy, relative to <source_dir>/<subject>/.", default="surf/lh.pial")
    parser.add_argument('-l', '--subjects-file', help="The subjects file to use. Text file with one subject per line. If left at default, <source_dir>/subjects.txt is used.", default="_")    
    parser.add_argument("-n", "--not-so-deep", help="Do not deep-copy, rename files to reflect origin and place all directly in target-dir instead.", action="store_true")
    parser.add_argument('-a', '--add-suffix', help="An optional suffix to add to the target file name.", default="")
    args = parser.parse_args()


    print("---Copy training/test data from a FreeSurfer recon-all output directory---")
    if args.verbose:
        print("Verbosity turned on.")
        print("Current working directory is '{cwd}'.".format(cwd=os.getcwd()))
        print("Using source data directory '{source_dir}' and target directory '{target_dir}'".format(source_dir=args.source_dir, target_dir=args.target_dir))

    if not os.path.isdir(args.source_dir):
        raise ValueError("The source directory '{source_dir}' does not exist or cannot be accessed".format(source_dir=args.source_dir))

    if not os.path.isdir(args.target_dir):
        raise ValueError("The target directory '{target_dir}' does not exist or cannot be accessed".format(target_dir=args.target_dir))

    subjects_file = args.subjects_file
    if subjects_file == "_":
        subjects_file = os.path.join(args.source_dir, 'subjects.txt')
    
    if args.verbose:
        print("Copying files from '{source_dir}' to '{target_dir}'.".format(source_dir=args.source_dir, target_dir=args.target_dir))
        print("Using subjects file '{subjects_file}'".format(subjects_file=subjects_file))

    if not os.path.isfile(subjects_file):
        raise ValueError("The file '{subjects_file}' does not exist or cannot be read".format(subjects_file=subjects_file))

    subjects_list = nit.read_subjects_file(subjects_file)

    files_per_subject = []
    if args.file:
        files_per_subject = [args.file]
        if args.verbose:
            print("Copying a single file per subject: '{sfile}'".format(sfile=args.file))
    else:
        files_per_subject = nit.read_subjects_file(args.file_list)    # It is not really a subjects file, but the format is the same.
        if args.verbose:
            print("Copying {num_files} files per subject.".format(num_files=len(files_per_subject)))

    num_files_total = len(files_per_subject)*len(subjects_list)
    if args.verbose:
        print("Copying {num_files_per_subject} files per subject for {num_subjects} subjects: {num_files_total} files in total.".format(num_files_per_subject=len(files_per_subject), num_subjects=len(subjects_list), num_files_total=num_files_total))

    if args.verbose:
        if args.not_so_deep:
            print("Using not-so-deep mode: copying all files directly into directory {target_dir} with adapted file names.".format(target_dir=args.target_dir))
        else:
            print("Using deep mode: copying all files into recon-all directory structure under {target_dir}.".format(target_dir=args.target_dir))

    num_files_failed = 0
    for subject in subjects_list:
        for sfile_rel in files_per_subject:
            source_file = os.path.join(args.source_dir, subject, sfile_rel)
            subject_rel_dir = os.path.dirname(os.path.join(subject, sfile_rel)) # required below for reconstruction/creation of target path.

            if args.not_so_deep:
                sfile_rel_no_dir_separator = sfile_rel.replace(os.sep, "_") # replace the OS-specific dir separator with an underscore.
                target_file_name = "{subject}_{sfile}".format(subject=subject, sfile=sfile_rel_no_dir_separator)
                dest_file = os.path.join(args.target_dir, target_file_name)
                dest_subdir = args.target_dir
            else:
                dest_file = os.path.join(args.target_dir, subject, sfile_rel)
                dest_subdir = os.path.join(args.target_dir, subject_rel_dir)

            if args.add_suffix:
                dest_file = "{dest_file}{suffix}".format(dest_file=dest_file,suffix=args.suffix)
                if args.verbose:
                    print("Added suffix {suffix}, dest file is now '{dest_file}'".format(suffix=args.suffix, dest_file=dest_file))

            if not os.path.isfile(source_file):
                print("WARNING: Expected source file '{source_file}' does not exist or cannot be read, skipping.".format(source_file=source_file))
                num_files_failed = num_files_failed + 1
            else:
                
                if not os.path.exists(dest_subdir):
                    os.makedirs(dest_subdir, exist_ok=True)
                try:
                    shutil.copyfile(source_file, dest_file)
                except:
                    print("WARNING: Failed to copy source file '{source_file}' to destination '{dest_file}'.".format(source_file=source_file, dest_file=dest_file))
                    num_files_failed = num_files_failed + 1
    
    print("Done. {num_failed} of {num_total} files failed ({num_ok} okay).".format(num_failed=num_files_failed, num_total=num_files_total, num_ok=num_files_total-num_files_failed))


if __name__ == "__main__":
    deepcopy_testdata_freesurfer()