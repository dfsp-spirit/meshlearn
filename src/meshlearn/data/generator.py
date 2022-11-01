# -*- coding: utf-8 -*-


from meshlearn.util.recon import get_valid_mesh_desc_file_pairs_reconall

"""
A data generator that yields data from a recon-all directory structure. Useful to train
models that can train in batches (like neural networks) with more data than fits into
your machine's memory.

This file is part of meshlearn, see https://github.com/dfsp-spirit/meshlearn for details.
"""


def neighborhood_generator_filepairs(batch_size, input_filepair_list, preproc_settings):
    """
    Generator for vertex neighborhood data from filepair list.

    Parameters
    ----------
    batch_size          : int, number of rows to return in each batch. Depends on model and your machine's memory (RAM).
    input_filepair_list : list of 2-tuples of type str, str. The first entry is a mesh file, the 2nd entry the respective pvd-data file.
    proproc_settings    : dict containing the preproc settings.
    """
    if not isinstance(input_filepair_list, list):
        raise ValueError("Parameter 'input_filepair_list' must be a list.")
    if len(input_filepair_list) < 1:
        raise ValueError("Parameter 'input_filepair_list' must not be empty, at least one input file pair is required.")
    while True: # We go for unlimited number of epochs. In each epoch, we load all available training data.
        # Fill pool for the first time.
        pair_idx = 0
        filepair = input_filepair_list[pair_idx]
        mesh_file, pvd_file = filepair
        neigh_pool = compute_dataset_for_mesh(mesh_file, preproc_settings, descriptor_file=pvd_file)
        pair_idx += 1

        print(f"Added first file '{mesh_file}' to pool, now at size {neigh_pool.shape[0]} neighborhoods.")

        while neigh_pool.shape[0] > 0:  # Iterate through all neighborhoods in all files.
            # If the neigh_pool is not full enough, fill it up.
            while neigh_pool.shape[0] < batch_size and pair_idx < len(input_filepair_list):
                filepair = input_filepair_list[pair_idx]
                mesh_file, pvd_file = filepair
                print(f"Pool contains {neigh_pool.shape[0]} neighborhoods, batch size is {batch_size}. Adding data from mesh file #{pair_idx} '{mesh_file}'.")
                df = compute_dataset_for_mesh(mesh_file, preproc_settings, descriptor_file=pvd_file)
                neigh_pool = neigh_pool.append(df, ignore_index=True)
                pair_idx += 1

            if pair_idx == len(input_filepair_list):
                print(f"No files left, pool contains {neigh_pool.shape[0]} neighborhoods now, batch size is {batch_size}.")
            else:
                print(f"Fill cycle done, pool contains {neigh_pool.shape[0]} neighborhoods now, batch size is {batch_size}.")

            # The pool is filled, we can return from it. We always return from the top, and remove the ones we returned from the pool.
            neigh_pool.reset_index(inplace=True, drop=True)
            start_index = 0
            end_index = batch_size if neigh_pool.shape[0] >= batch_size else neigh_pool.shape[0]  # There may only be less left at the end.
            batch_df = neigh_pool.iloc[start_index:end_index]
            neigh_pool.drop(neigh_pool.index[range(start_index, end_index)], axis=0, inplace=True) # Remove used ones from the top.
            neigh_pool.reset_index(inplace=True, drop=True)

            print(f"Returning batch_df with top {batch_df.shape[0]} neighborhoods, {neigh_pool.shape[0]} left in pool.")

            nc = len(batch_df.columns)
            labels = batch_df[:, (nc-1)]
            descriptors = batch_df[:, 0:(nc-1)]
            yield descriptors, labels



def neighborhood_generator_reconall_dir(batch_size, data_settings, preproc_settings):
    """
    Generator for vertex neighborhood data from mesh files in recon-all output directory.

    Parameters
    ----------
    batch_size        : int, number of rows to return in each batch. Depends on model and your machine's memory (RAM).
    data_settings     : dict containing the data settings, like 'data_dir', 'surface', and 'descriptor'.
    proproc_settings  : dict containing the preproc settings.
    """
    if not isinstance(data_settings, dict):
        raise ValueError("Parameter 'data_settings' must be a dict.")
    mesh_files, desc_files, cortex_files, files_subject, files_hemi, miss_subjects = get_valid_mesh_desc_file_pairs_reconall(data_settings['data_dir'], surface=data_settings['surface'], descriptor=data_settings['descriptor'], cortex_label=preproc_settings.get('cortex_label', False), verbose=data_settings.get("verbose", True), subjects_file=data_settings.get('subjects_file', None), subjects_list=data_settings.get('subjects_list', None), hemis=data_settings.get('hemis', ["lh", "rh"]))
    input_filepair_list = list(zip(mesh_files, desc_files))
    return neighborhood_generator_filepairs(batch_size, input_filepair_list, preproc_settings)
