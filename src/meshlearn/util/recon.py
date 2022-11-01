# -*- coding: utf-8 -*-

"""
Utility functions related to the recon-all directory structure. The 'recon-all' program is the
main FreeSurfer program for preprocessing sMRI data, and stores the output for a all subjects
of a study in a special directory structure, typicalle referred to as 'SUBJECTS_DIR' (which is
also the name of the environment variable used to specify it for FreeSurfer commands). These
functions help with dealing with such a dir structure.

This file is part of meshlearn, see https://github.com/dfsp-spirit/meshlearn for details.
"""


import brainload.nitools as nit
import numpy as np
import os.path
import glob


def get_valid_mesh_desc_file_pairs_reconall(recon_dir, surface="pial", descriptor="pial_lgi", verbose=True, subjects_file=None, subjects_list=None, hemis=["lh", "rh"], cortex_label=False):
    """
    Discover valid pairs of mesh and descriptor files in FreeSurfer recon-all output dir.

    Parameters
    ----------
    recon_dir     : str, recon-all output dir.
    surface       : str, surface file to load. 'white', 'pial', 'sphere', etc
    descriptor    : str or None, desc to load (per-vertex data). Something like 'thickness', 'volume', 'area', or 'None' for no desciptor.
    verbose       : bool, whether to print status info
    subjects_file : str or None, path to subjects text file with one subject per line (CSV without header, with one column). Assumed to be `recon_dir/subjects.txt` if omitted and `subjects_list` is also `None`. Use this or `subjects_list`, not both.
    subjects_list : list of str, None, or 'autodetect'. Defines the subjects to load. If a list of str, these are interpreted as subject names (sub directories if the `recon_dir`). If the str `autodetect`, valid sub directories under `recon_dir` are auto-detected and their names are used as the subjects list. Used only if no `subjects_file` is given. Use this or `subjects_file`, not both.
    hemis         : list of str, containing one or both of 'lh', 'rh'. Can be used to get only files for one hemisphere.
    cortex_label  : bool, whether to also require `label/<hemi>.cortex.label` files for the subjects, and return them in `valid_labl_files` return value.

    Returns
    -------
    valid_mesh_files           : list of str, filenames of valid mesh files. Valid means that the other required files for this mesh files also exist (the `descriptor_file` and the `cortex_label` file, if requested).
    valid_desc_files           : list of str, filenames of descriptor files for the valid_mesh_files. If not requested, a list of None.
    valid_labl_files           : list of str, filenames of cortex label files for the valid_mesh_files. If not requested, a list of None.
    valid_files_subject        : list of str, for all valid_mesh_files, the subject to which the mesh belongs.
    valid_files_hemi           : list of str, for all valid_mesh_files, the hemisphere to which the mesh belongs ('lh' or 'rh').
    subjects_missing_some_file : list of str, the subject names that were missing some files. Note that subjects can occur twice, if both their hemis were requested and some file was missing for each hemisphere.

    See also
    --------
    get_valid_mesh_desc_lgi_file_pairs_flat_dir: similar function that works with a flattened input dir. Prefer this recon-all version.
    """
    if not os.path.isdir(recon_dir):
        raise ValueError(f"The data directory '{recon_dir}' does not exist or cannot be accessed")

    if subjects_file is not None and subjects_list is not None:
        raise ValueError("Pass only one of 'subjects_file' and 'subjects_list', not both.")

    if isinstance(subjects_list, str):
        if subjects_list == "autodetect":
            subjects_list = nit.detect_subjects_in_directory(recon_dir)
        else:
            raise ValueError(f"Invalid 'subjects_list' parameter. Must be None, a list of str, or 'autodetect' but found single, invalid string '{subjects_list}'.")

    if subjects_list is None:
        if subjects_file is None:  # Assume standard subjects file in data dir.
            subjects_file = os.path.join(recon_dir, "subjects.txt")
            print(f"INFO: No 'subjects_list' and no 'subjects_file' given for loading data, assuming subjects file '{subjects_file}'.")
        if not os.path.isfile(subjects_file):
            raise ValueError(f"Subjects file '{subjects_file}' cannot be read.")
        subjects_list = nit.read_subjects_file(subjects_file)

    if verbose:
        print(f"Using subjects list containing {len(subjects_list)} subjects. Loading them from recon-all output dir '{recon_dir}'.")
        if descriptor is None:
            print(f"Discovering surface '{surface}' for {len(hemis)} hemis: {hemis}.")
        else:
            print(f"Discovering surface '{surface}', descriptor '{descriptor}' for {len(hemis)} hemis: {hemis}.")
        if cortex_label:
            print(f"Discovering cortex labels.")
        else:
            print(f"Not discovering cortex labels.")

    valid_mesh_files = []  # Mesh files (one per hemi)
    valid_desc_files = []  # per-vertex descriptor files (one per hemi), like lGI
    valid_labl_files = []  # cortex label files, if requested.
    valid_files_hemi = []     # For each entry in the previous 3 lists, the hemisphere ('lh' or 'rh') to which the files belong.
    valid_files_subject = [] # For each entry in the previous 3 lists, the subject to which the files belong.
    subjects_missing_some_file = [] # All subjects which were missing one or more of the requested files. No data from them gets returned.

    for subject in subjects_list:
        sjd = os.path.join(recon_dir, subject)
        if os.path.isdir(sjd):
            for hemi in hemis:
                surf_file = os.path.join(sjd, "surf", f"{hemi}.{surface}")
                desc_file = os.path.join(sjd, "surf", f"{hemi}.{descriptor}")

                if cortex_label:
                    labl_file = os.path.join(sjd, "label", f"{hemi}.cortex.label")
                    if descriptor is None:
                        if os.path.isfile(surf_file) and os.path.isfile(labl_file):
                            valid_mesh_files.append(surf_file)
                            valid_desc_files.append(None)
                            valid_labl_files.append(labl_file)
                            valid_files_subject.append(subject)
                            valid_files_hemi.append(hemi)
                        else:
                            subjects_missing_some_file.append(subject)
                    else:
                        if os.path.isfile(surf_file) and os.path.isfile(desc_file) and os.path.isfile(labl_file):
                            valid_mesh_files.append(surf_file)
                            valid_desc_files.append(desc_file)
                            valid_labl_files.append(labl_file)
                            valid_files_subject.append(subject)
                            valid_files_hemi.append(hemi)
                        else:
                            subjects_missing_some_file.append(subject)

                else:
                    if descriptor is None:
                        if os.path.isfile(surf_file):
                            valid_mesh_files.append(surf_file)
                            valid_desc_files.append(None)
                            valid_files_subject.append(subject)
                            valid_files_hemi.append(hemi)
                        else:
                            subjects_missing_some_file.append(subject)
                    else:
                        if os.path.isfile(surf_file) and os.path.isfile(desc_file):
                            valid_mesh_files.append(surf_file)
                            valid_desc_files.append(desc_file)
                            valid_files_subject.append(subject)
                            valid_files_hemi.append(hemi)
                        else:
                            subjects_missing_some_file.append(subject)

    if verbose:
        cortex_tag = ""
        if cortex_label:
            cortex_tag = " and cortex label"

        if descriptor is None:
            print(f"Out of {len(subjects_list)*2} subject hemispheres ({len(subjects_list)} subjects), {len(valid_mesh_files)} had the requested surface{cortex_tag} files.")
        else:
            print(f"Out of {len(subjects_list)*2} subject hemispheres ({len(subjects_list)} subjects), {len(valid_mesh_files)} had the requested surface and descriptor{cortex_tag} files.")
        if len(subjects_missing_some_file) > 0:
            print(f"The following {len(subjects_missing_some_file)} subjects where missing files: {', '.join(subjects_missing_some_file)}")

    return valid_mesh_files, valid_desc_files, valid_labl_files, valid_files_subject, valid_files_hemi, subjects_missing_some_file





def get_valid_mesh_desc_lgi_file_pairs_flat_dir(dc_data_dir, verbose=True):
    """
    Discover valid pairs of mesh and descriptor files in datadir created with `deepcopy_testdata.py`  script and the `--not-so-deep` option.

    WARNING: Note that `dc_data_dir` is NOT a standard FreeSurfer directory structure, but a flat directory with
            renamed files (including subject to make them unique in the dir). Use the mentioned script `deepcopy_testdata.py` with the
            `--not-so-deep` command line option to
            turn a FreeSUrfer recon-all output dir into such a flat dir.

    Parameters
    -----------
    dc_data_dir: str, heavily modified (flattened) recon-all output dir structure: Flat directory with
                      renamed files (including subject to make them unique in the dir). Use the mentioned
                      script `deepcopy_testdata.py` with the `--not-so-deep` command line option to
                      turn a FreeSUrfer recon-all output dir into such a flat dir.
    verbose: bool, whether to print verbose output.

    See also
    --------
    get_valid_mesh_desc_file_pairs_reconall: similar function that works with a standard recon-all output dir. Prefer that.

    Returns
    -------
    tuple of 2 lists of filenames, the first list is a list of pial surface mesh files. the 2nd a list of lgi descriptor files. It is
    guaranteed that the lists have some lengths, and that the files at identical indices in them belong to each other.
    """

    if not os.path.isdir(dc_data_dir):
        raise ValueError("The data directory '{data_dir}' does not exist or cannot be accessed".format(data_dir=dc_data_dir))

    mesh_files = np.sort(glob.glob("{data_dir}/*.pial".format(data_dir=dc_data_dir)))
    descriptor_files = np.sort(glob.glob("{data_dir}/*.pial_lgi".format(data_dir=dc_data_dir)))
    if verbose:
        if len(mesh_files) < 3:
            print("Found {num_mesh_files} mesh files: {mesh_files}".format(num_mesh_files=len(mesh_files), mesh_files=', '.join(mesh_files)))
        else:
            print("Found {num_mesh_files} mesh files, first 3: {mesh_files}".format(num_mesh_files=len(mesh_files), mesh_files=', '.join(mesh_files[0:3])))
        if len(descriptor_files) < 3:
            print("Found {num_descriptor_files} descriptor files: {descriptor_files}".format(num_descriptor_files=len(descriptor_files), descriptor_files=', '.join(descriptor_files)))
        else:
            print("Found {num_descriptor_files} descriptor files, first 3: {descriptor_files}".format(num_descriptor_files=len(descriptor_files), descriptor_files=', '.join(descriptor_files[0:3])))

    valid_mesh_files = list()
    valid_desc_files = list()

    for mesh_filename in mesh_files:
        expected_desc_filename = "{mesh_filename}_lgi".format(mesh_filename=mesh_filename)
        if os.path.exists(expected_desc_filename):
            valid_mesh_files.append(mesh_filename)
            valid_desc_files.append(expected_desc_filename)

    assert len(valid_mesh_files) == len(valid_desc_files)
    num_valid_file_pairs = len(valid_mesh_files)

    if verbose:
        print("Found {num_valid_file_pairs} valid pairs of mesh file with matching descriptor file.".format(num_valid_file_pairs=num_valid_file_pairs))
    return valid_mesh_files, valid_desc_files
