# -*- coding: utf-8 -*-

"""
Read FreeSurfer brain meshes and pre-computed lgi per-vertex data for them from a directory
and perform mesh pre-processing (including computation of global and local mesh descriptors)
on them to create a full dataset.

This file is part of meshlearn, see https://github.com/dfsp-spirit/meshlearn for details.
"""

import trimesh as tm
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from meshlearn.data.mem_opt import reduce_mem_usage
from meshlearn.data.neighborhood import neighborhoods_euclid_around_points
from meshlearn.util.recon import get_valid_mesh_desc_file_pairs_reconall

from warnings import warn
import os.path
import time
import psutil
import json
from datetime import timedelta
from sys import getsizeof
import random


class TrainingData():

    @staticmethod
    def data_from_files(mesh_file_name, descriptor_file_name):
        """
        Extract mesh and descriptor data from a single pair of files.

        Parameters
        ----------
        mesh_file_name: str, the mesh file name. Must be a FreeSurfer surf file.
        descriptor_file_name: str or None, the descriptor file name. Must be FreeSurfer curv file that assigns one value to each vertex of the mesh_file_name. If None, the returned `pvd_data` will be None.

        Returns
        -------
        3-tuple of:
        vert_coords: 2d nx3 float ndarray of n vertex coordinates in 3D space
        faces: 2d nx3 integer ndarray of n triangles, given as indices into the vert_coords
        pvd_data: 1d n float ndarray or None, one value per vertex of per-vertex data. If `descriptor_file_name` is `None`, this is `None`.
        """
        vert_coords, faces = fsio.read_geometry(mesh_file_name)
        pvd_data = None
        if descriptor_file_name is not None:
            pvd_data = fsio.read_morph_data(descriptor_file_name)
        return vert_coords, faces, pvd_data


    def neighborhoods_from_raw_data_parallel(self, datafiles, mesh_neighborhood_radius, mesh_neighborhood_count, exactly=False, num_samples_per_file=None, df=True, verbose=False, add_desc_vertex_index=False, add_desc_neigh_size=False, num_cores=8, num_files_total=None, filter_smaller_neighborhoods=False, add_desc_brain_bbox=True, add_subject_and_hemi_columns=False, reduce_mem=True, random_seed=None, add_local_mesh_descriptors = True, add_global_mesh_descriptors = True):
        """
        Parallel version of `neighborhoods_from_raw_data`. Calls the latter in parallel using multi-threading.

        Note that this does not take the `num_samples_total` parameter, as this cannot be respected when loading files in parallel. Use a combination of `num_samples_per_file`/`exactly` and `num_files_total` instead to limit the dataset to a size compatible with your available RAM.

        Parameters
        ----------
        datafiles                    : list of 2-tuples str, 1st elem of each tuple: str, mesh file name. 2nd elem: str, corresponding per-vertex data file name. Must be FreeSurfer surf files and curv files.
        mesh_neighborhood_radius     : radius for neighborhood sphere, in mesh units (mm for FreeSurfer meshes)
        exactly                      : bool, whether to force loading exactly 'num_samples_total' samples. If false, and the last chunk loaded from a file leads to more samples, this function will return all loaded ones. If true, the extra ones will be discarded and exactly 'num_samples_total' samples will be returned.
        num_samples_per_file         : positive integer, the number of samples (neighborhoods) to load at max per mesh file. Can be used to read data from more different subjects, while still keeping the total training data size reasonable. Note that the function may return less, if filtering by size is active via `max_num_neighbors`.
        df                           : bool, whether to return as pandas.DataFrame (instead of numpy.ndarray). Internal, leave alone, or subsequent functions may fail.
        verbose                      : bool, whether to print output (or be silent)
        mesh_neighborhood_count      : int or None, number of neighbors to consider at most per vertex (even if more were found within the `mesh_neighborhood_radius` during kdtree search). Set to None for all.
        add_desc_vertex_index        : bool, whether to add descriptor: vertex index in mesh
        add_desc_neigh_size          : bool, whether to add descriptor: number of neighbors in ball query radius (before any filtering due to `mesh_neighborhood_count`)
        num_cores                    : int, number of cores to use for parallel data loading.
        num_files_total              : int or None, number of files to load. Set to None for all. Consider your available RAM.
        filter_smaller_neighborhoods : bool, whether to skip neighborhoods smaller than `mesh_neighborhood_count`. If false, missing vertex values are filled with NAN.
        add_desc_brain_bbox          : bool, whether to add descriptor: brain bounding box
        add_subject_and_hemi_columns : bool whether to add extra columns contains subject ID and hemi.
        reduce_mem                   : bool, whether to convert all datatypes to memory saving version with lower precision to save RAM (e.g., int64 to in32), if the value range allows it.
        random_seed                  : int or None, seed for random number generator.
        add_local_mesh_descriptors   : bool, whether to add local mesh descriptors, like local curvature.
        add_global_mesh_descriptors  : bool, whether to add global mesh descriptors, like vertex and edge counts.

        Note: Parameter 'num_samples_total' is not supported in parallel mode.
              Use 'num_files_total' (and 'num_samples_per_file') instead
              to limit total number of entries."
        """
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        if not isinstance(datafiles, list):
            raise ValueError(f"[par] datafiles must be a list of 2-tuples, but is not a list: {type(datafiles)}")
        if len(datafiles) == 0:
            raise ValueError(f"[par] datafiles must not be empty")
        if not isinstance(datafiles[0], tuple):
            raise ValueError(f"[par] datafiles must be a list of 2-tuples, it is a list but does not contain tuples: {type(datafiles[0])}.")

        if num_files_total is not None:
            if num_files_total < len(datafiles):
                datafiles_subset = datafiles[0:num_files_total]
                assert len(datafiles_subset) == num_files_total
                datafiles = datafiles_subset

        with ThreadPoolExecutor(num_cores) as pool:
            neighborhoods_from_raw_single_file_pair = partial(self.neighborhoods_from_raw_data_seq, mesh_neighborhood_radius=mesh_neighborhood_radius, num_samples_total=None, exactly=exactly, num_samples_per_file=num_samples_per_file, df=df, verbose=verbose, mesh_neighborhood_count=mesh_neighborhood_count, add_desc_vertex_index=add_desc_vertex_index, add_desc_neigh_size=add_desc_neigh_size, filter_smaller_neighborhoods=filter_smaller_neighborhoods, add_desc_brain_bbox=add_desc_brain_bbox, add_subject_and_hemi_columns=add_subject_and_hemi_columns, reduce_mem=reduce_mem, random_seed=random_seed, add_local_mesh_descriptors=add_local_mesh_descriptors, add_global_mesh_descriptors=add_global_mesh_descriptors)
            df = pd.concat(pool.map(neighborhoods_from_raw_single_file_pair, datafiles))
        return df, df.columns, datafiles


    def neighborhoods_from_raw_data_seq(self, datafiles, mesh_neighborhood_radius, mesh_neighborhood_count, num_samples_total=None, exactly=False, num_samples_per_file=None, df=True, verbose=True, add_desc_vertex_index=False, add_desc_neigh_size=False, filter_smaller_neighborhoods=False, add_desc_brain_bbox=True, add_subject_and_hemi_columns=False, reduce_mem=True, random_seed=None, add_local_mesh_descriptors = True, add_global_mesh_descriptors = True):
        """Loader for training data from FreeSurfer format (non-preprocessed) files, also does the preprocessing on the fly.

        Will load mesh and descriptor files, and use a kdtree to quickly find, for each vertex, all neighbors withing Euclidean distance 'neighborhood_radius'.
        Returns, for each such vertex neighborhood, the coordinates and normals of all neighborhood vertices.
        Note that the data must fit into memory. Use this or `gen_data`, depending on whether or not you want everything in memory at once.

        Use the `num_samples_total` parameter, or a combination of `num_samples_per_file`/`exactly` and `num_files_total` to limit the dataset to a size compatible with your available RAM.

        Parameters
        ----------
        datafiles                    : list of 2-tuples str, 1st elem of each tuple: str, mesh file name. 2nd elem: str, corresponding per-vertex data file name. Must be FreeSurfer surf files and curv files.
        mesh_neighborhood_radius     : radius for neighborhood sphere, in mesh units (mm for FreeSurfer meshes)
        num_samples_total            : positive integer, the total number of samples (neighborhoods) to return from the mesh files. Set to None to return all values. A sample consists of the data for a single vertex, i.e., its neighborhood coordinates and its target per-vertex value. Setting to None is slower, because we cannot pre-allocate.
        exactly                      : bool, whether to force loading exactly 'num_samples_total' samples. If false, and the last chunk loaded from a file leads to more samples, this function will return all loaded ones. If true, the extra ones will be discarded and exactly 'num_samples_total' samples will be returned.
        num_samples_per_file         : positive integer, the number of samples (neighborhoods) to load at max per mesh file. Can be used to read data from more different subjects, while still keeping the total training data size reasonable. Note that the function may return less, if filtering by size is active via `max_num_neighbors`.
        df                           : bool, whether to return as pandas.DataFrame (instead of numpy.ndarray)
        verbose                      : bool, whether to print output (or be silent)
        mesh_neighborhood_count      : int, number of neighbors to consider at most per vertex (even if more were found within the `mesh_neighborhood_radius` during kdtree search).
        add_desc_vertex_index        : bool, whether to add descriptor: vertex index in mesh
        add_desc_neigh_size          : bool, whether to add descriptor: number of neighbors in ball query radius (before any filtering due to `mesh_neighborhood_count`)
        filter_smaller_neighborhoods : bool, whether to skip neighborhoods smaller than `mesh_neighborhood_count`. If false, missing vertex values are filled with NAN.
        add_desc_brain_bbox          : bool, whether to add descriptor: brain bounding box
        add_subject_and_hemi_columns : bool whether to add extra columns contains subject ID and hemi.
        reduce_mem                   : bool, whether to convert all datatypes to memory saving version with lower precision to save RAM (e.g., int64 to in32), if the value range allows it.
        random_seed                  : int or None, seed for random number generator.
        add_local_mesh_descriptors   : bool, whether to add local mesh descriptors, like local curvature.
        add_global_mesh_descriptors  : bool, whether to add global mesh descriptors, like vertex and edge counts.


        Returns
        ------
        X 2d nx3 float np.ndarray of neighborhood coordinates, each row contains the x,y,z coords of a single vertex. The n rows form the neighborhood around the source vertex.
        y scalar float, the per-vertex data value for the source vertex.
        """
        if mesh_neighborhood_radius is None:
            raise ValueError("Must pass non-None value for parameter 'neighborhood_radius'.")

        if mesh_neighborhood_count is None:
            raise ValueError("Must pass non-None value for parameter 'max_num_neighbors'.")

        is_parallel_wrapped = False
        if isinstance(datafiles, tuple):  # We come from the parallel wrapper 'neighborhoods_from_raw_data_parallel', and received a single tuple from the full list.
            if len(datafiles) == 2:
                datafiles_tmp = list()
                datafiles_tmp.append(datafiles)
                datafiles = datafiles_tmp   # We wrap this into a list (with 1 element) because the sequential function works with a list.
                if verbose:
                    print(f"[seq, wrapped in parallel] Wrapping tuple ({datafiles[0][0]}, {datafiles[0][1]},) into list.")
                is_parallel_wrapped = True
            else:
                raise ValueError(f"[seq] Received tuple (assuming parallel mode) with length {len(datafiles)}, but required is length 2.")

        if not isinstance(datafiles, list):
            raise ValueError(f"[seq] datafiles must be a list of 2-tuples, but is not a list: {type(datafiles)}")
        if len(datafiles) == 0:
            raise ValueError(f"[seq] datafiles must not be empty")
        if not isinstance(datafiles[0], tuple):
            raise ValueError(f"[seq] datafiles must be a list of tuples, it is a list but does not contain tuples: {type(datafiles[0])}.")

        num_samples_loaded = 0
        do_break = False
        full_data = None

        num_files_loaded = 0
        datafiles_loaded = []
        if verbose:
                print(f"[load] Loading data.")
        for filepair in datafiles:
            subject = None
            hemi = None
            if len(filepair) == 2:
                mesh_file_name, descriptor_file_name = filepair
            elif len(filepair) == 4:
                mesh_file_name, descriptor_file_name, subject, hemi = filepair
            if do_break:
                break

            if not os.path.exists(mesh_file_name) and os.path.exists(descriptor_file_name):
                warn("[load] Skipping non-existant file pair '{mf}' and '{df}'.".format(mf=mesh_file_name, df=descriptor_file_name))
                continue

            if verbose:
                print(f"[load] * Loading mesh file '{mesh_file_name}' and descriptor file '{descriptor_file_name}'.")
            vert_coords, faces, pvd_data = TrainingData.data_from_files(mesh_file_name, descriptor_file_name)
            assert faces.ndim == 2
            datafiles_loaded.append(filepair)

            num_verts_total = vert_coords.shape[0]

            if num_samples_per_file == None:
                query_vert_coords = vert_coords
                query_vert_indices = np.arange(num_verts_total)
            else:
                query_vert_coords = vert_coords.copy()
                # Sample 'num_samples_per_file' vertex coords from the full coords list
                randomstate = np.random.default_rng(random_seed)
                query_vert_indices = randomstate.choice(num_verts_total, num_samples_per_file, replace=False, shuffle=False)
                query_vert_coords = query_vert_coords[query_vert_indices, :]

            mesh = tm.Trimesh(vertices=vert_coords, faces=faces)

            compute_extra_columns_time_start = time.time()

            extra_columns = {}
            # Add extreme coords of the brain (min and max for each axis), as a proxy for total brain size.
            if add_desc_brain_bbox:
                assert vert_coords.ndim == 2
                assert vert_coords.shape[1] == 3
                ones = np.ones((num_verts_total), dtype=np.float32)
                extra_columns['xmin'] = ones * np.min(vert_coords[:, 0])
                extra_columns['xmax'] = ones * np.max(vert_coords[:, 0])
                extra_columns['ymin'] = ones * np.min(vert_coords[:, 1])
                extra_columns['ymax'] = ones * np.max(vert_coords[:, 1])
                extra_columns['zmin'] = ones * np.min(vert_coords[:, 2])
                extra_columns['zmax'] = ones * np.max(vert_coords[:, 2])

            if add_subject_and_hemi_columns:
                if subject is None or hemi is None:
                    raise ValueError("Parameter 'add_subject_and_hemi_columns' is True, but cannot add subject and hemi columns, required data not supplied in 'datafiles' tuples.")
                else:
                    extra_columns['subject'] = np.array([subject] * num_verts_total)
                    extra_columns['hemi'] = np.array([hemi] * num_verts_total)

            if add_local_mesh_descriptors:
                from meshlearn.data.curvature import Curvature
                c = Curvature(mesh_file_name)
                descriptors_to_compute = ["gaussian_curvature", "mean_curvature", "shape_index", "curvedness_index"]
                shape_desc = c.compute(descriptors_to_compute)
                for desc in descriptors_to_compute:
                    extra_columns[desc] = shape_desc[[desc]].to_numpy()

            if add_global_mesh_descriptors:
                x_extend = np.max(vert_coords[:, 0]) - np.min(vert_coords[:, 0])
                y_extend = np.max(vert_coords[:, 1]) - np.min(vert_coords[:, 1])
                z_extend = np.max(vert_coords[:, 2]) - np.min(vert_coords[:, 2])
                bb_vol= x_extend * y_extend * z_extend
                ones = np.ones((num_verts_total), dtype=np.float32)
                extra_columns['aabb_vol'] = ones * bb_vol  # Enclosed volume (volume of aabb, the axis-aligned bounding box)
                extra_columns['area'] = ones * mesh.area  # Mesh area
                extra_columns['ctr_mass_x'] = ones * mesh.center_mass[0]  # Mesh center of mass, x coordinate.
                extra_columns['ctr_mass_x'] = ones * mesh.center_mass[1]  # Mesh center of mass, y coordinate.
                extra_columns['ctr_mass_x'] = ones * mesh.center_mass[2]  # Mesh center of mass, z coordinate.
                extra_columns['num_edges'] = ones * mesh.edges.shape[0]  # Number of edges in the mesh
                extra_columns['mean_edge_len'] = ones * np.float32(np.mean(mesh.edges_unique_length))  # Mean edge length
                extra_columns['volume'] = ones * mesh.volume  # Mesh volume
                extra_columns['num_faces'] = ones * faces.shape[0]  # Number of faces

            compute_extra_columns_time_end = time.time()
            compute_extra_columns_execution_time = compute_extra_columns_time_end - compute_extra_columns_time_start
            if verbose:
                print(f"[load] Adding {len(extra_columns)} extra descriptor columns for current file done, it took: {timedelta(seconds=compute_extra_columns_execution_time)}.")


            if verbose:
                print(f"[load]  - Computing neighborhoods based on radius {mesh_neighborhood_radius} for {query_vert_coords.shape[0]} of {num_verts_total} vertices in mesh file '{mesh_file_name}'.")

            neighborhoods, col_names, kept_vertex_indices_mesh = neighborhoods_euclid_around_points(query_vert_coords, query_vert_indices, KDTree(vert_coords), neighborhood_radius=mesh_neighborhood_radius, mesh=mesh, max_num_neighbors=mesh_neighborhood_count, pvd_data=pvd_data, add_desc_vertex_index=add_desc_vertex_index, add_desc_neigh_size=add_desc_neigh_size, verbose=verbose, filter_smaller_neighborhoods=filter_smaller_neighborhoods, extra_columns=extra_columns)

            num_files_loaded += 1

            neighborhoods_size_bytes = getsizeof(neighborhoods)
            if verbose:
                print(f"[load]  - Current {neighborhoods.shape[0]} neighborhoods (from file #{num_files_loaded}) size in RAM is about {int(neighborhoods_size_bytes / 1024. / 1024.)} MB.")

            if full_data is None:
                full_data = neighborhoods
            else:
                full_data = np.concatenate((full_data, neighborhoods,), axis=0)
                full_data_size_bytes = getsizeof(full_data)
                full_data_size_MB = int(full_data_size_bytes / 1024. / 1024.)
                if verbose:
                    print(f"[load]  - Currently after {num_files_loaded} files, {full_data.shape[0]} neighborhoods loaded, and full_data size in RAM is about {full_data_size_MB} MB ({int(full_data_size_MB / num_files_loaded)} MB per file on avg). {int(psutil.virtual_memory().available / 1024. / 1024.)} MB RAM still available.")

            num_samples_loaded += neighborhoods.shape[0]

            if num_samples_total is not None:
                if num_samples_loaded >= num_samples_total:
                        if verbose:
                            print(f"[load] Done loading the requested {num_samples_total} samples, ignoring the rest.")
                        do_break = True
                        break

        if num_samples_total is not None:
                if num_samples_loaded > num_samples_total:
                    if exactly:
                        if verbose:
                             print(f"[load] Truncating data of size {num_samples_loaded} to {num_samples_total} samples, 'exactly' is true.")
                        full_data = full_data[0:num_samples_total, :] # this wastes stuff we spent time loading
                    else:
                        if verbose:
                            print(f"[load] Returning {num_samples_loaded} instead of {num_samples_total} samples, file contained more and 'exactly' is false.")


        if df:
            full_data = pd.DataFrame(full_data, columns=col_names)
            if verbose:
                dataset_size_bytes = full_data.memory_usage(deep=True).sum()
                print(f"[load] Total dataset size in RAM is about {int(dataset_size_bytes / 1024. / 1024.)} MB.")
            if reduce_mem:
                full_data = reduce_mem_usage(full_data)
                if verbose:
                    dataset_size_bytes = full_data.memory_usage(deep=True).sum()
                    print(f"[load] Total dataset size in RAM after memory optimization is about {int(dataset_size_bytes / 1024. / 1024.)} MB.")
            if verbose:
                print(f"[load] RAM available is about {int(psutil.virtual_memory().available / 1024. / 1024.)} MB")

        if is_parallel_wrapped:
            return full_data
        else:
            return full_data, col_names, datafiles_loaded



def compute_dataset_for_mesh(mesh_file, preproc_settings, descriptor_file=None, verbose=False, data_settings = {'num_samples_total': None,
                     'num_samples_per_file': None,
                     'random_seed': None,
                     'exactly': False,
                     'reduce_mem': False  # Reducing mem takes a lot of time, is not needed here and we want fast predictions.
                     }):
    """
    Perform loading and pre-processing for a single mesh file. Useful when predicting.

    Parameters
    ----------
    mesh_file        : str, the mesh to load.
    descriptor_file  : str or None, the per-vertex descriptor file to load. Can be None if you only want to load and pre-process the mesh, without adding a label column. If you supply it, the last column of the returned DataFrame will hold the labels read from this file.
    preproc_settings : dict, the pre-processing settings for the mesh. Must match those used for the model when predicting, get them from the model JSON file.
    verbose          : bool, whether to print output.
    data_settings    : dict, the data settings which are related to which samples are loaded, but do not change descriptors (i.e., they affect row, but not columns in the dataset). Leave alone for prediction, as people want to predict for the whole mesh anyways.
    """
    input_filepair_list = [(mesh_file, descriptor_file, )]
    settings_out = {'data_settings': data_settings, 'preproc_settings': preproc_settings, 'log': dict()}

    preproc_settings.pop('cortex_label', None)



    load_start = time.time()
    tdl = TrainingData()
    dataset, col_names, datafiles_loaded = tdl.neighborhoods_from_raw_data_seq(input_filepair_list,
                                                                               **data_settings,
                                                                               **preproc_settings)
    load_end = time.time()
    load_execution_time = load_end - load_start
    if verbose:
        print(f"=== Loading data file done, it took: {timedelta(seconds=load_execution_time)} ===")

    settings_out['log']['datafiles_loaded'] = datafiles_loaded

    assert isinstance(dataset, pd.DataFrame)
    return dataset, col_names, settings_out


def compute_dataset_from_datadir(data_settings, preproc_settings):
    """
    Very high-level wrapper with debug info around `Trainingdata.neighborhoods_from_raw_data_seq` and `Trainingdata.neighborhoods_from_raw_data_parallel`.

    Parameters
    -----------
    data_settings: dict with keys:
        data_dir: str, recon-all output dir.
        surface: str or mesh, FreeSurfer surface mesh available for subjects in data_dir. Something like `white` or `pial`.
        descriptor: str or None, FreeSurfer per-vertex descriptor available for subjects in data_dir. Something like `thickness` or `pial_lgi`, or `area`. If None, no descriptor data will be loaded (useful if you load to predict).
        sequential: whether to load data sequentially (as opposed to parallel).
        verbose: bool, whether to print verbose output.
        num_neighborhoods_to_load: int or None, the total number of neighborhoods (rows) to load. Set to None for all in the files. Only considered if `sequential=True`.
        num_samples_per_file: int or None, the number of neighborhoods (rows) to load per file. See also `exactly`.
        exactly: bool, see `num_samples_per_file`: whether to disallow loading more if the file contained more.
        num_cores: int, number of CPU cores to use for parallel loading. depends on your cpu count and harddisk speed. ignored if `sequential=True`.
        num_files_to_load: int or None, number of files to load from those available in data_dir
        shuffle_input_file_order: bool, whether to shuffle input files before starting to load from the beginning.
        random_seed: None or int, something to seed the random number generators used for randomness in the functions. Set to None for random, or an int for a reproducible choice (e.g., which of the vertices to load, which files to load, etc.).
    preproc_settings: dict with keys:
        cortex_label: bool, whether to load cortex label file from recon-all dir. Not implementend yet, must be False.
        add_desc_vertex_index: bool, whether to add descriptor: vertex index in mesh
        add_desc_neigh_size: bool, whether to add descriptor: number of neighbors in ball query radius (before any filtering due to `mesh_neighborhood_count`)
        mesh_neighborhood_radius: float, radius of sphere for kdtree ball point query (neighbor search).
        mesh_neighborhood_count: int, number of neighbors to consider at most per vertex (even if more were found within the mesh_neighborhood_radius during kdtree search).
        filter_smaller_neighborhoods: bool, whether to skip neighborhoods smaller than `mesh_neighborhood_count`. If false, missing vertex values are filled with NAN.
        add_desc_brain_bbox: whether to add descriptor: brain bounding box
        add_subject_and_hemi_columns: bool whether to add extra columns containing subject identifier and hemi.
        add_local_mesh_descriptors   : bool, whether to add local mesh descriptors, like local curvature.
        add_global_mesh_descriptors  : bool, whether to add global mesh descriptors, like vertex and edge counts.
    """
    if preproc_settings.get('cortex_label', False):
        raise ValueError("Parameter preproc_settings['cortex_label'] must be False: not implemented yet.")
    settings_out = {'data_settings': data_settings, 'preproc_settings': preproc_settings, 'log': dict()}
    discover_start = time.time()
    mesh_files, desc_files, cortex_files, files_subject, files_hemi, miss_subjects = get_valid_mesh_desc_file_pairs_reconall(data_settings['data_dir'], surface=data_settings['surface'], descriptor=data_settings['descriptor'], cortex_label=preproc_settings['cortex_label'])

    verbose = data_settings['verbose']

    assert len(mesh_files) == len(desc_files)
    assert len(mesh_files) == len(files_hemi)
    if preproc_settings.get('cortex_label', False):
        assert len(cortex_files) == len(mesh_files)
    else:
        assert len(cortex_files) == 0
    assert len(mesh_files) == len(files_subject)

    ## These are not of great interest, as the files listed here were not necessarily loaded.
    ## Still, this can be interesting to see which subjects are missing data files.
    ## See 'datafiles_loaded' below, which is more relevant.
    log_available_data = False
    if log_available_data:
        settings_out['log']['datadir_available_mesh_files'] = mesh_files        # contains all mesh files of subjects (subject hemispheres, to be precise) which had all required files.
        settings_out['log']['datadir_available_desc_files'] = desc_files        # contains all descriptor files of subjects (subject hemispheres, to be precise) which had all required files.
        settings_out['log']['datadir_available_cortex_files'] = cortex_files    # if 'cortex_label' is True, contains all cortex.label files of subjects (subject hemispheres, to be precise) which had all required files.
        settings_out['log']['datadir_available_files_subject'] = files_subject    # The subject for each of the returned files (in the order in which the files appear above).
        settings_out['log']['datadir_available_files_hemi'] = files_hemi    # The hemis ('lh' or 'rh') for each of the returned valid files.
        settings_out['log']['datadir_available_miss_subjects'] = miss_subjects  # Subjects that are missing one or more of the requested files. They were ignored, and none of their files show up in mesh_files, desc_files (and cortex_files if requested).

    discover_end = time.time()
    discover_execution_time = discover_end - discover_start
    if verbose:
        print(f"=== Discovering data files done, it took: {timedelta(seconds=discover_execution_time)} ===")

    # Adding a subject and hemi column allows users to filter by subject or hemi later, but a better way
    # is to have them provide an input directory that only contains the files they want, so this option
    # is not exposed to users currently, and turned off (because we do not want the model to learn from the
    # subject name and hemisphere at this time).
    add_subject_and_hemi_columns = False
    if add_subject_and_hemi_columns:
        input_filepair_list = list(zip(mesh_files, desc_files, files_subject, files_hemi))  # List of 4-tuples, for each tuple first elem is mesh_file, 2nd is desc_file, 3rd is source subject, 4th is source hemi ('lh' or 'rh').
    else:
        input_filepair_list = list(zip(mesh_files, desc_files))  # List of 2-tuples, for each tuple first elem is mesh_file, 2nd is desc_file.

    # Shuffle input file list if requested. Useful to ensure that we do not handle only the first X files, which are all from the same site.

    if data_settings.get('shuffle_input_file_order', False):
        if verbose:
            print(f"Shuffling input file list with random seed '{data_settings.get('random_seed', None)}'.")
        random.seed(data_settings.get('random_seed', None))
        random.shuffle(input_filepair_list)
    else:
        if verbose:
            print(f"Not shuffling input file list.")

    num_cores_tag = "all" if data_settings['num_cores'] is None or data_settings['num_cores'] == 0 else data_settings['num_cores']
    seq_par_tag = " sequentially " if data_settings['sequential'] else f" in parallel using {num_cores_tag} cores"

    if verbose:
        print(f"Discovered {len(input_filepair_list)} valid pairs of input mesh and descriptor files.")

        if data_settings['sequential']:
            if data_settings['num_neighborhoods_to_load'] is None:
                print(f"Will load all data from the {len(input_filepair_list)} files{seq_par_tag}.")
            else:
                print(f"Will load {data_settings['num_neighborhoods_to_load']} samples in total from the {len(input_filepair_list)} files.")
        else:
            if data_settings['num_files_to_load'] is None:
                print(f"Will load data from all {len(input_filepair_list)} files{seq_par_tag}.")
            else:
                print(f"Will load data from {data_settings['num_files_to_load']} input files.")

        if data_settings['num_samples_per_file'] is None:
            print(f"Will load all suitably sized vertex neighborhoods from each mesh file.")
        else:
            print(f"Will load at most {data_settings['num_samples_per_file']} vertex neighborhoods per mesh file.")

        neigh_size_tag = "auto-determined neighborhood size" if preproc_settings['mesh_neighborhood_count'] is None or preproc_settings['mesh_neighborhood_count'] == 0 else f"neighborhood size {preproc_settings['mesh_neighborhood_count']}"
        if preproc_settings.get('filter_smaller_neighborhoods', False):
            print(f"Will filter (remove) all neighborhoods smaller than {neigh_size_tag}.")
        else:
            print(f"NOTICE: Will fill the respective missing columns of neighborhoods smaller than {neigh_size_tag} with NAN values. You will have to handle NAN values before training! (Set 'filter_smaller_neighborhoods' to 'True' to ignore them instead.)")


    load_start = time.time()
    tdl = TrainingData()

    add_local_mesh_descriptors = preproc_settings.get("add_local_mesh_descriptors", True)
    add_global_mesh_descriptors = preproc_settings.get("add_global_mesh_descriptors", True)

    if data_settings['sequential']:
        dataset, col_names, datafiles_loaded = tdl.neighborhoods_from_raw_data_seq(input_filepair_list, mesh_neighborhood_radius=preproc_settings['mesh_neighborhood_radius'], mesh_neighborhood_count=preproc_settings['mesh_neighborhood_count'], num_samples_total=data_settings['num_neighborhoods_to_load'], num_samples_per_file=data_settings['num_samples_per_file'], add_desc_vertex_index=preproc_settings['add_desc_vertex_index'], add_desc_neigh_size=preproc_settings['add_desc_neigh_size'], filter_smaller_neighborhoods=preproc_settings['filter_smaller_neighborhoods'], exactly=data_settings['exactly'], add_desc_brain_bbox=preproc_settings['add_desc_brain_bbox'], add_subject_and_hemi_columns=add_subject_and_hemi_columns, random_seed=data_settings.get('random_seed', None), add_local_mesh_descriptors=add_local_mesh_descriptors, add_global_mesh_descriptors=add_global_mesh_descriptors)
    else:
        dataset, col_names, datafiles_loaded = tdl.neighborhoods_from_raw_data_parallel(input_filepair_list, mesh_neighborhood_radius=preproc_settings['mesh_neighborhood_radius'], mesh_neighborhood_count=preproc_settings['mesh_neighborhood_count'], num_files_total=data_settings['num_files_to_load'], num_samples_per_file=data_settings['num_samples_per_file'], add_desc_vertex_index=preproc_settings['add_desc_vertex_index'], add_desc_neigh_size=preproc_settings['add_desc_neigh_size'], num_cores=data_settings['num_cores'], filter_smaller_neighborhoods=preproc_settings['filter_smaller_neighborhoods'], exactly=data_settings['exactly'], add_desc_brain_bbox=preproc_settings['add_desc_brain_bbox'], add_subject_and_hemi_columns=add_subject_and_hemi_columns, random_seed=data_settings.get('random_seed', None), add_local_mesh_descriptors=add_local_mesh_descriptors, add_global_mesh_descriptors=add_global_mesh_descriptors)
    load_end = time.time()
    load_execution_time = load_end - load_start
    if verbose:
        print(f"=== Loading data files{seq_par_tag} done, it took: {timedelta(seconds=load_execution_time)} ===")

    settings_out['log']['datafiles_loaded'] = datafiles_loaded

    assert isinstance(dataset, pd.DataFrame)
    return dataset, col_names, settings_out



def get_dataset_pickle(data_settings_in, preproc_settings, do_pickle_data, dataset_pickle_file=None, dataset_settings_file=None, verbose=True):
    """
    Wrapper around `compute_dataset` that additionally uses pickling if requested.

    If `do_pickle_data` is `True` and the file `dataset_pickle_file` exists, it will load the dataset from the given pkl file, ignoring the `data_settings_in`.
    Otherwise, it will compute the dataset from the dataset (load raw mesh + descriptor files and compute mesh neighborhoods from them), using the settings from `data_settings_in`.

    Parameters
    ----------
    data_settings_in: dict, passed on as kwargs to compute_dataset if we do not load a pickled dataset. Ignored otherwise.
    do_pickle_data: bool, whether to pickle data (compute and then save if no file found, load if found.)
    dataset_pickle_file: str, the pkl file to load the pickled dataset from, or save it to if not exists. Ignored if do_pickle_data=False.
    dataset_settings_file: str, the JSON file to load the dataset metadata from, or save it to if not exists. Ignored if do_pickle_data=False.
    verbose: bool, whether to print verbose info
    """
    if not isinstance(data_settings_in, dict):
        raise ValueError(f"Parameter 'data_settings_in' must be a dictionary.")
    if not isinstance(preproc_settings, dict):
        raise ValueError(f"Parameter 'preproc_settings' must be a dictionary.")

    if do_pickle_data and (dataset_pickle_file is None or dataset_settings_file is None):
        raise ValueError(f"If 'do_pickle_data' is 'True', a valid 'dataset_pickle_file' and 'dataset_settings_file' have to be supplied.")

    if do_pickle_data and os.path.isfile(dataset_pickle_file):
        pickle_file_size_mb = int(os.path.getsize(dataset_pickle_file) / 1024. / 1024.)
        if verbose:
            print("==========================================================================================================================================================================")
            print(f"WARNING: Unpickling pre-saved dataframe from {pickle_file_size_mb} MB pickle file '{dataset_pickle_file}', ignoring all dataset settings! Delete file or set 'do_pickle_data' to False to prevent.")
            print("==========================================================================================================================================================================")
        unpickle_start = time.time()
        dataset = pd.read_pickle(dataset_pickle_file)
        col_names = dataset.columns
        if verbose:
            unpickle_end = time.time()
            pickle_load_time = unpickle_end - unpickle_start
            print(f"INFO: Loaded dataset with shape {dataset.shape} from pickle file '{dataset_pickle_file}'. It took {timedelta(seconds=pickle_load_time)}.")
        try:
            with open(dataset_settings_file, 'r') as fp:
                data_settings = json.load(fp)
                if verbose:
                    print(f"INFO: Loaded settings used to create dataset from file '{dataset_settings_file}'.")
        except Exception as ex:
            data_settings = None
            if verbose:
                print(f"NOTICE: Could not load settings used to create dataset from file '{dataset_settings_file}': {str(ex)}.")
    else:
        dataset, col_names, data_settings = compute_dataset_from_datadir(data_settings_in, preproc_settings)
        if do_pickle_data:
            pickle_start = time.time()
            # Save the settings as a JSON file.
            with open(dataset_settings_file, 'w') as fp:
                json.dump(data_settings, fp, sort_keys=True, indent=4)
            # Save the dataset itself as a pkl file.
            dataset.to_pickle(dataset_pickle_file)
            if verbose:
                pickle_end = time.time()
                pickle_save_time = pickle_end - pickle_start
                pickle_file_size_mb = int(os.path.getsize(dataset_pickle_file) / 1024. / 1024.)
                print(f"INFO: Saved dataset to pickle file '{dataset_pickle_file}' ({pickle_file_size_mb} MB) and dataset settings to '{dataset_settings_file}', ready to load next run. Saving dataset took {timedelta(seconds=pickle_save_time)}.")
    return dataset, col_names, data_settings


