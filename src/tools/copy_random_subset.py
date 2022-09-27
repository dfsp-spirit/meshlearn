#!/usr/bin/env python

# Copies a subset of the ABIDE subjects, and for them only the requested descriptors, to a new dir.

import os

import brainload.nitools as nit

def copy_blah(subjects_dir, subjects_file=None, subjects_list=None):
    if not os.path.isdir(subjects_dir):
        raise ValueError("The data directory '{subjects_dir}' does not exist or cannot be accessed".format(data_dir=subjects_dir))

    if subjects_file is not None and subjects_list is not None:
        raise ValueError("Pass only one of 'subjects_file' and 'subjects_list', not both.")

    if subjects_file is None and subjects_list is None: # Assume standard subjects file in data dir.
        subjects_file = os.path.join(subjects_dir, "subjects.txt")

    if not subjects_file is None:
        if not os.path.isfile(subjects_file):
            raise ValueError(f"Subjects file '{subjects_file}' cannot be read.")
        subjects_list = nit.read_subjects_file(subjects_file)

    list_subfolders_with_paths = [f.path for f in os.scandir(subjects_dir) if f.is_dir()]
