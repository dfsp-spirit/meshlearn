# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import psutil

from meshlearn.util.recon import get_valid_mesh_desc_file_pairs_reconall
from meshlearn.data.postproc import postproc_settings
from meshlearn.data.training_data import compute_dataset_for_mesh

"""
A data generator that yields data from a recon-all directory structure. Useful to train
models that can train in batches (like neural networks) with more data than fits into
your machine's memory.

This file is part of meshlearn, see https://github.com/dfsp-spirit/meshlearn for details.
"""


def neighborhood_generator_filepairs(batch_size, input_filepair_list, preproc_settings, verbose=False):
    """
    Generator for vertex neighborhood data from filepair list.

    This keeps a reservoir of neighborhoods (in a `pd.DataFrame`), and only loads and pre-processes a new mesh file
    if a new batch is requested and and the reservoir has less entries that the current `batch_size`.

    This could be improved to use async-io.

    Parameters
    ----------
    batch_size          : int, number of rows to return in each batch. Depends on model and your machine's memory (RAM).
    input_filepair_list : list of 2-tuples of type str, str. The first entry is a mesh file, the 2nd entry the respective pvd-data file. Must not be empty.
    proproc_settings    : dict containing the preproc settings.

    Returns
    -------
    desciptors: 2D NxM np.ndarray, containing N observations of M descriptors.
    labels: 1D np.ndarray of size N, the labels for the observations.
    """
    if not isinstance(input_filepair_list, list):
        raise ValueError("Parameter 'input_filepair_list' must be a list.")
    if len(input_filepair_list) < 1:
        raise ValueError("Parameter 'input_filepair_list' must not be empty, at least one input file pair is required.")

    do_replace_nan = postproc_settings['do_replace_nan']
    replace_nan_with = postproc_settings['replace_nan_with']
    do_scale_descriptors = postproc_settings['do_scale_descriptors']
    scale_func = postproc_settings['scale_func']

    while True: # We go for unlimited number of epochs. In each epoch, we load all available training data.
        # Fill pool for the first time.
        pair_idx = 0
        filepair = input_filepair_list[pair_idx]
        mesh_file, pvd_file = filepair
        neigh_pool, col_names, settings_out = compute_dataset_for_mesh(mesh_file, preproc_settings, descriptor_file=pvd_file)
        pair_idx += 1

        if verbose:
            print(f"[gen] Added first file '{mesh_file}' to pool, now at size {neigh_pool.shape[0]} neighborhoods.")

        while neigh_pool.shape[0] > 0:  # Iterate through all neighborhoods in all files.
            # If the neigh_pool is not full enough, fill it up.
            while neigh_pool.shape[0] < batch_size and pair_idx < len(input_filepair_list):
                filepair = input_filepair_list[pair_idx]
                mesh_file, pvd_file = filepair
                if verbose:
                    print(f"[gen] Pool contains {neigh_pool.shape[0]} neighborhoods, batch size is {batch_size}. Adding data from mesh file #{pair_idx} '{mesh_file}'.")
                df, col_names, settings_out = compute_dataset_for_mesh(mesh_file, preproc_settings, descriptor_file=pvd_file)
                #neigh_pool = neigh_pool.append(df, ignore_index=True)
                neigh_pool = pd.concat([neigh_pool, df], axis=0, ignore_index=True)
                pair_idx += 1

            if pair_idx == len(input_filepair_list):
                if verbose:
                    print(f"[gen] No files left, pool contains {neigh_pool.shape[0]} neighborhoods now, batch size is {batch_size}.")
            else:
                if verbose:
                    print(f"[gen] Fill cycle done, pool contains {neigh_pool.shape[0]} neighborhoods now, batch size is {batch_size}.")

            # The pool is filled, we can return from it. We always return from the top, and remove the ones we returned from the pool.
            neigh_pool.reset_index(inplace=True, drop=True)

            start_index = 0
            end_index = batch_size if neigh_pool.shape[0] >= batch_size else neigh_pool.shape[0]  # There may only be less left at the end.
            batch_df = neigh_pool.iloc[start_index:end_index]
            assert isinstance(batch_df, pd.DataFrame)
            assert len(batch_df.shape) == 2, f"batch_df has shape {batch_df.shape} with len {len(batch_df.shape)}, expected len 2."
            neigh_pool.drop(neigh_pool.index[start_index:end_index,], inplace=True) # Remove used rows from the top.
            neigh_pool.reset_index(inplace=True, drop=True)

            if verbose:
                print(f"[gen] Returning batch_df with top {batch_df.shape[0]} neighborhoods, {neigh_pool.shape[0]} left in pool.")

            # Handle NaN values.
            if do_replace_nan:
                row_indices_with_nan_values = pd.isnull(batch_df).any(1).to_numpy().nonzero()[0]
                if row_indices_with_nan_values.size > 0:
                    if verbose:
                        print(f"[gen] Batch contains {row_indices_with_nan_values.size} rows (observations) with NAN values (of {batch_df.shape[0]} observations total).")
                    batch_df = batch_df.fillna(replace_nan_with, inplace=False)
                    if verbose:
                        print(f"[gen] Replacing NAN values in {row_indices_with_nan_values.size} columns with value '{replace_nan_with}'.")
                    row_indices_with_nan_values = pd.isnull(batch_df).any(1).to_numpy().nonzero()[0]
                    if verbose:
                        print(f"[gen] Batch contains {row_indices_with_nan_values.size} rows (observations) with NAN values (of {batch_df.shape[0]} observations total) after filling. {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left.")
                else:
                    if verbose:
                        print(f"[gen] Batch contains no NAN values. {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM left. Batch shape is {batch_df.shape}.")
                del row_indices_with_nan_values

            assert isinstance(batch_df, pd.DataFrame)
            assert len(batch_df.shape) == 2, f"batch_df has shape {batch_df.shape} with len {len(batch_df.shape)}, expected len 2."
            nc = len(batch_df.columns)
            if verbose:
                print(f"[gen] Batch shape is {batch_df.shape}, with {nc} columns.")
            labels = batch_df.iloc[:, (nc-1)].to_numpy()
            descriptors = batch_df.iloc[:, 0:(nc-1)]
            if do_scale_descriptors:
                with np.errstate(divide='ignore'):  # Ignore run-time warnings about division by zero, we fix the resulting NAN values (if any) below.
                    descriptors = scale_func(descriptors.to_numpy())
                descriptors[np.isnan(descriptors)] = replace_nan_with

            yield descriptors, labels



def neighborhood_generator_reconall_dir(batch_size, data_settings, preproc_settings, verbose=None):
    """
    Generator for vertex neighborhood data from mesh files in recon-all output directory.

    Parameters
    ----------
    batch_size        : int, number of rows to return in each batch. Depends on model and your machine's memory (RAM).
    data_settings     : dict containing the data settings, like 'data_dir', 'surface', and 'descriptor'.
    proproc_settings  : dict containing the preproc settings.
    verbose           : bool or None, whether to print verbose output. If None, the boolean `data_settings['verbose']` is used, and if it does not exist, the fallback value is `False`.

    Returns
    -------
    desciptors: 2D NxM np.ndarray, containing N observations of M descriptors.
    labels: 1D np.ndarray of size N, the labels for the observations.
    """
    if not isinstance(data_settings, dict):
        raise ValueError("Parameter 'data_settings' must be a dict.")
    if verbose is None:
        verbose = data_settings.get('verbose', False)
    mesh_files, desc_files, cortex_files, files_subject, files_hemi, miss_subjects = get_valid_mesh_desc_file_pairs_reconall(data_settings['data_dir'], surface=data_settings['surface'], descriptor=data_settings['descriptor'], cortex_label=preproc_settings.get('cortex_label', False), verbose=verbose, subjects_file=data_settings.get('subjects_file', None), subjects_list=data_settings.get('subjects_list', None), hemis=data_settings.get('hemis', ["lh", "rh"]))
    input_filepair_list = list(zip(mesh_files, desc_files))
    return neighborhood_generator_filepairs(batch_size, input_filepair_list, preproc_settings, verbose=verbose)
