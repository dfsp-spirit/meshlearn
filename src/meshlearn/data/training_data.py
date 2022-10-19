# -*- coding: utf-8 -*-

"""
Read FreeSurfer brain meshes and pre-computed lgi per-vertex data for them from a directory.

This file is part of meshlearn, see https://github.com/dfsp-spirit/meshlearn for details.
"""

import trimesh as tm
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from meshlearn.data.mem_opt import reduce_mem_usage
from meshlearn.data.neighborhood import neighborhoods_euclid_around_points
from warnings import warn
import os.path
import glob
import time
import psutil
import json
from datetime import timedelta
from sys import getsizeof
import random


class TrainingData():

    def __init__(self, neighborhood_radius, num_neighbors, allow_nan=False):
        """
        Parameters
        ----------
        datafiles: dict str,str of mesh_file_name : pvd_file_name
        neighborhood_radius: The ball radius for kdtree ball queries, defining the neighborhood.
        num_neighbors: int, the number of neighbors to actually use per vertex (keeps only num_neighbors per neighborhood), to fix the dimension of the descriptor. Each vertex must have at least this many neighbors for this to work, unless allow_nan is True.
        allow_nan: boolean, whether to continue in case a neighborhood is smaller than num_neighbors. If this is True and the situation occurs, np.nan values will be added as neighbor coordinates. If this is False and the situation occurs, an error is raised. Whether or not allowing nans makes sense depends on the machine learning method used downstream. Many methods cannot handle nan values.
        """
        self.neighborhood_radius = neighborhood_radius
        self.num_neighbors = num_neighbors
        self.allow_nan = allow_nan

        self.kdtree = None  # Do not use in multi-threading context if sharing the TrainingData instance between threads.
        self.mesh = None    # Do not use in multi-threading context if sharing the TrainingData instance between threads.

    @staticmethod
    def data_from_files(mesh_file_name, descriptor_file_name):
        """
        Extract mesh and descriptor data from a single pair of files.

        Parameters
        ----------
        mesh_file_name: str, the mesh file name. Must be a FreeSurfer surf file.
        descriptor_file_name: str, the descriptor file name. Must be FreeSurfer curv file that assigns one value to each vertex of the mesh_file_name.

        Returns
        -------
        3-tuple of:
        vert_coords: 2d nx3 float ndarray of n vertex coordinates in 3D space
        faces: 2d nx3 integer ndarray of n triangles, given as indices into the vert_coords
        pvd_data: 1d n float ndarray, one value per vertex of per-vertex data
        """
        vert_coords, faces = fsio.read_geometry(mesh_file_name)
        pvd_data = fsio.read_morph_data(descriptor_file_name)
        return vert_coords, faces, pvd_data


    def neighborhoods_from_raw_data_parallel(self, datafiles, neighborhood_radius=None, exactly=False, num_samples_per_file=None, df=True, verbose=False, max_num_neighbors=None, add_desc_vertex_index=False, add_desc_neigh_size=False, num_cores=8, num_files_total=None, filter_smaller_neighborhoods=False, add_desc_brain_bbox=True, add_subject_and_hemi_columns=False, reduce_mem=True, random_seed=None):
        """
        Parallel version of `neighborhoods_from_raw_data`. Calls the latter in parallel using multi-threading.

        Note that this does not take the `num_samples_total` parameter, as this cannot be respected when loading files in parallel. Use a combination of `num_samples_per_file`/`exactly` and `num_files_total` instead to limit the dataset to a size compatible with your available RAM.

        Parameters
        ----------
        datafiles: list of 2-tuples str, 1st elem of each tuple: str, mesh file name. 2nd elem: str, corresponding per-vertex data file name. Must be FreeSurfer surf files and curv files.
        neighborhood_radius: radius for neighborhood sphere, in mesh units (mm for FreeSurfer meshes)
        exactly: bool, whether to force loading exactly 'num_samples_total' samples. If false, and the last chunk loaded from a file leads to more samples, this function will return all loaded ones. If true, the extra ones will be discarded and exactly 'num_samples_total' samples will be returned.
        num_samples_per_file: positive integer, the number of samples (neighborhoods) to load at max per mesh file. Can be used to read data from more different subjects, while still keeping the total training data size reasonable. Note that the function may return less, if filtering by size is active via `max_num_neighbors`.
        df : bool, whether to return as pandas.DataFrame (instead of numpy.ndarray)
        verbose: bool, whether to print output (or be silent)
        max_num_neighbors: int or None, number of neighbors to consider at most per vertex (even if more were found within the mesh_neighborhood_radius during kdtree search). Set to None for all.
        add_desc_vertex_index: bool, whether to add descriptor: vertex index in mesh
        add_desc_neigh_size: bool, whether to add descriptor: number of neighbors in ball query radius (before any filtering due to `mesh_neighborhood_count`)
        num_cores: int, number of cores to use for parallel data loading.
        num_files_total: int or None, number of files to load. Set to None for all. Consider your available RAM.
        filter_smaller_neighborhoods: bool, whether to skip neighborhoods smaller than `mesh_neighborhood_count`. If false, missing vertex values are filled with NAN.
        add_desc_brain_bbox: whether to add descriptor: brain bounding box
        add_subject_and_hemi_columns: bool whether to add extra columns contains subject ID and hemi.
        reduce_mem: bool, whether to convert all datatypes to memory saving version with lower precision to save RAM (e.g., int64 to in32), if the value range allows it.
        random_seed: int or None, seed for random number generator.

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
            neighborhoods_from_raw_single_file_pair = partial(self.neighborhoods_from_raw_data_seq, neighborhood_radius=neighborhood_radius, num_samples_total=None, exactly=exactly, num_samples_per_file=num_samples_per_file, df=df, verbose=verbose, max_num_neighbors=max_num_neighbors, add_desc_vertex_index=add_desc_vertex_index, add_desc_neigh_size=add_desc_neigh_size, filter_smaller_neighborhoods=filter_smaller_neighborhoods, add_desc_brain_bbox=add_desc_brain_bbox, add_subject_and_hemi_columns=add_subject_and_hemi_columns, reduce_mem=reduce_mem, random_seed=random_seed)
            df = pd.concat(pool.map(neighborhoods_from_raw_single_file_pair, datafiles))
        return df, df.columns, datafiles


    def neighborhoods_from_raw_data_seq(self, datafiles, neighborhood_radius=None, num_samples_total=None, exactly=False, num_samples_per_file=None, df=True, verbose=True, max_num_neighbors=None, add_desc_vertex_index=False, add_desc_neigh_size=False, filter_smaller_neighborhoods=False, add_desc_brain_bbox=True, add_subject_and_hemi_columns=False, reduce_mem=True, random_seed=None):
        """Loader for training data from FreeSurfer format (non-preprocessed) files, also does the preprocessing on the fly.

        Will load mesh and descriptor files, and use a kdtree to quickly find, for each vertex, all neighbors withing Euclidean distance 'neighborhood_radius'.
        Returns, for each such vertex neighborhood, the coordinates and normals of all neighborhood vertices.
        Note that the data must fit into memory. Use this or `gen_data`, depending on whether or not you want everything in memory at once.

        Use the `num_samples_total` parameter, or a combination of `num_samples_per_file`/`exactly` and `num_files_total` to limit the dataset to a size compatible with your available RAM.

        Parameters
        ----------
        datafiles: list of 2-tuples str, 1st elem of each tuple: str, mesh file name. 2nd elem: str, corresponding per-vertex data file name. Must be FreeSurfer surf files and curv files.
        neighborhood_radius: radius for neighborhood sphere, in mesh units (mm for FreeSurfer meshes)
        num_samples_total: positive integer, the total number of samples (neighborhoods) to return from the mesh files. Set to None to return all values. A sample consists of the data for a single vertex, i.e., its neighborhood coordinates and its target per-vertex value. Setting to None is slower, because we cannot pre-allocate.
        exactly: bool, whether to force loading exactly 'num_samples_total' samples. If false, and the last chunk loaded from a file leads to more samples, this function will return all loaded ones. If true, the extra ones will be discarded and exactly 'num_samples_total' samples will be returned.
        num_samples_per_file: positive integer, the number of samples (neighborhoods) to load at max per mesh file. Can be used to read data from more different subjects, while still keeping the total training data size reasonable. Note that the function may return less, if filtering by size is active via `max_num_neighbors`.
        df : bool, whether to return as pandas.DataFrame (instead of numpy.ndarray)
        verbose: bool, whether to print output (or be silent)
        max_num_neighbors: int, number of neighbors to consider at most per vertex (even if more were found within the mesh_neighborhood_radius during kdtree search).
        add_desc_vertex_index: bool, whether to add descriptor: vertex index in mesh
        add_desc_neigh_size: bool, whether to add descriptor: number of neighbors in ball query radius (before any filtering due to `mesh_neighborhood_count`)
        filter_smaller_neighborhoods: bool, whether to skip neighborhoods smaller than `mesh_neighborhood_count`. If false, missing vertex values are filled with NAN.
        add_desc_brain_bbox: whether to add descriptor: brain bounding box
        add_subject_and_hemi_columns: bool whether to add extra columns contains subject ID and hemi.
        reduce_mem: bool, whether to convert all datatypes to memory saving version with lower precision to save RAM (e.g., int64 to in32), if the value range allows it.
        random_seed: int or None, seed for random number generator.


        Returns
        ------
        X 2d nx3 float np.ndarray of neighborhood coordinates, each row contains the x,y,z coords of a single vertex. The n rows form the neighborhood around the source vertex.
        y scalar float, the per-vertex data value for the source vertex.
        """
        if neighborhood_radius is None:
            neighborhood_radius = self.neighborhood_radius

        if max_num_neighbors is None:
            max_num_neighbors = self.num_neighbors

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
            self.mesh = None  # Cannot use self.mesh due to required thread-safety.

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

            add_local_mesh_descriptors = True  # TODO: expose as function parameter
            add_global_mesh_descriptors = True  # TODO: expose as function parameter

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


            self.kdtree = None # Cannot use self.kdtree due to required thread-safety.
            if verbose:
                print(f"[load]  - Computing neighborhoods based on radius {neighborhood_radius} for {query_vert_coords.shape[0]} of {num_verts_total} vertices in mesh file '{mesh_file_name}'.")

            neighborhoods, col_names, kept_vertex_indices_mesh = neighborhoods_euclid_around_points(query_vert_coords, query_vert_indices, KDTree(vert_coords), neighborhood_radius=neighborhood_radius, mesh=mesh, max_num_neighbors=max_num_neighbors, pvd_data=pvd_data, add_desc_vertex_index=add_desc_vertex_index, add_desc_neigh_size=add_desc_neigh_size, verbose=verbose, filter_smaller_neighborhoods=filter_smaller_neighborhoods, extra_columns=extra_columns)

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



def get_valid_mesh_desc_file_pairs_reconall(recon_dir, surface="pial", descriptor="pial_lgi", verbose=True, subjects_file=None, subjects_list=None, hemis=["lh", "rh"], cortex_label=False):
    """
    Discover valid pairs of mesh and descriptor files in FreeSurfer recon-all output dir.

    Parameters
    ----------
    recon_dir str, recon-all output dir.
    surface str, surface file to load. 'white', 'pial', 'sphere', etc
    descriptor str, desc to load (per-vertex data). 'thickness', 'volume', 'area' etc
    verbose bool, whether to print status info
    subjects_file str, path to subjects file. assumed to be recon_dir/subjects.txt if omitted
    subjects_list list of str, use only if no subjects_file is given.
    hemis list of str, containing one or both of 'lh', 'rh'
    cortex_label bool, whether to also require label/<hemi>.cortex.label files.

    See also
    --------
    get_valid_mesh_desc_lgi_file_pairs_flat_dir: similar function that works with a flattened input dir. Prefer this recon-all version.
    """
    if not os.path.isdir(recon_dir):
        raise ValueError(f"The data directory '{recon_dir}' does not exist or cannot be accessed")

    if subjects_file is not None and subjects_list is not None:
        raise ValueError("Pass only one of 'subjects_file' and 'subjects_list', not both.")

    if subjects_file is None and subjects_list is None: # Assume standard subjects file in data dir.
        subjects_file = os.path.join(recon_dir, "subjects.txt")

    if not subjects_file is None:
        if not os.path.isfile(subjects_file):
            raise ValueError(f"Subjects file '{subjects_file}' cannot be read.")
        subjects_list = nit.read_subjects_file(subjects_file)

    if verbose:
        print(f"Using subjects list containing {len(subjects_list)} subjects. Loading them from recon-all output dir '{recon_dir}'.")
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
                    if os.path.isfile(surf_file) and os.path.isfile(desc_file) and os.path.isfile(labl_file):
                        valid_mesh_files.append(surf_file)
                        valid_desc_files.append(desc_file)
                        valid_labl_files.append(labl_file)
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
        print(f"Out of {len(subjects_list)*2} subject hemispheres ({len(subjects_list)} subjects), {len(valid_mesh_files)} had the requested surface and descriptor files.")
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


def compute_dataset(data_dir, surface="pial", descriptor="pial_lgi", cortex_label=False, verbose=False, num_neighborhoods_to_load=None, num_samples_per_file=None, add_desc_vertex_index=False, add_desc_neigh_size=False, sequential=False, num_cores=8, num_files_to_load=None, mesh_neighborhood_radius=10, mesh_neighborhood_count=300, filter_smaller_neighborhoods=False, exactly=False, add_desc_brain_bbox=True, add_subject_and_hemi_columns=False, shuffle_input_file_order=True, random_seed=None):
    """
    Very high-level wrapper with debug info around `Trainingdata.neighborhoods_from_raw_data_seq` and `Trainingdata.neighborhoods_from_raw_data_parallel`.

    Parameters
    -----------
    data_dir: str, recon-all output dir.
    surface: str, FreeSurfer surface mesh available for subjects in data_dir. Something like `white` or `pial`.
    descriptor: str, FreeSurfer per-vertex descriptor available for subjects in data_dir. Something like `thickness` or `pial_lgi`, or `area`.
    cortex_label: bool, whether to load cortex label file from recon-all dir. Not implementend yet.
    verbose: bool, whether to print verbose output.
    num_neighborhoods_to_load: int or None, the total number of neighborhoods (rows) to load. Set to None for all in the files. Only considered if `sequential=True`.
    num_samples_per_file: int or None, the number of neighborhoods (rows) to load per file. See also `exactly`.
    add_desc_vertex_index: bool, whether to add descriptor: vertex index in mesh
    add_desc_neigh_size: bool, whether to add descriptor: number of neighbors in ball query radius (before any filtering due to `mesh_neighborhood_count`)
    sequential: whether to load data sequentially (as opposed to parallel).
    num_cores: int, number of CPU cores to use for parallel loading. depends on your cpu count and harddisk speed. ignored if `sequential=True`.
    num_files_to_load: int or None, number of files to load from those available in data_dir
    mesh_neighborhood_radius: float, radius of sphere for kdtree ball point query (neighbor search).
    mesh_neighborhood_count: int, number of neighbors to consider at most per vertex (even if more were found within the mesh_neighborhood_radius during kdtree search).
    filter_smaller_neighborhoods: bool, whether to skip neighborhoods smaller than `mesh_neighborhood_count`. If false, missing vertex values are filled with NAN.
    exactly: bool, see `num_samples_per_file`: whether to disallow loading more if the file contained more.
    add_desc_brain_bbox: whether to add descriptor: brain bounding bo
    add_subject_and_hemi_columns: bool whether to add extra columns contains subject ID and hemi.
    shuffle_input_file_order: bool, whether to shuffle input files before starting to load from the beginning.
    random_seed: None or int, something to seed the random number generators used for randomness in the functions. Set to None for random, or an int for a reproducible choice (e.g., which of the vertices to load, which files to load, etc.).
    """
    if cortex_label:
        raise ValueError("Parameter 'cortex_label' must be False: not implemented yet.")
    data_settings = locals() # Capute passed parameters as dict.
    discover_start = time.time()
    mesh_files, desc_files, cortex_files, files_subject, files_hemi, miss_subjects = get_valid_mesh_desc_file_pairs_reconall(data_dir, surface=surface, descriptor=descriptor, cortex_label=cortex_label)

    assert len(mesh_files) == len(desc_files)
    assert len(mesh_files) == len(files_hemi)
    if cortex_label:
        assert len(cortex_files) == len(mesh_files)
    else:
        assert len(cortex_files) == 0
    assert len(mesh_files) == len(files_subject)

    ## These are not of great interest, as the files listed here were not necessarily loaded.
    ## Still, this can be interesting to see which subjects are missing data files.
    ## See 'datafiles_loaded' below, which is more relevant.
    log_available_data = False
    if log_available_data:
        data_settings['datadir_available_mesh_files'] = mesh_files        # contains all mesh files of subjects (subject hemispheres, to be precise) which had all required files.
        data_settings['datadir_available_desc_files'] = desc_files        # contains all descriptor files of subjects (subject hemispheres, to be precise) which had all required files.
        data_settings['datadir_available_cortex_files'] = cortex_files    # if 'cortex_label' is True, contains all cortex.label files of subjects (subject hemispheres, to be precise) which had all required files.
        data_settings['datadir_available_files_subject'] = files_subject    # The subject for each of the returned files (in the order in which the files appear above).
        data_settings['datadir_available_files_hemi'] = files_hemi    # The hemis ('lh' or 'rh') for each of the returned valid files.
        data_settings['datadir_available_miss_subjects'] = miss_subjects  # Subjects that are missing one or more of the requested files. They were ignored, and none of their files show up in mesh_files, desc_files (and cortex_files if requested).

    discover_end = time.time()
    discover_execution_time = discover_end - discover_start
    if verbose:
        print(f"=== Discovering data files done, it took: {timedelta(seconds=discover_execution_time)} ===")

    if add_subject_and_hemi_columns:
        input_filepair_list = list(zip(mesh_files, desc_files, files_subject, files_hemi))  # List of 4-tuples, for each tuple first elem is mesh_file, 2nd is desc_file, 3rd is source subject, 4th is source hemi ('lh' or 'rh').
    else:
        input_filepair_list = list(zip(mesh_files, desc_files))  # List of 2-tuples, for each tuple first elem is mesh_file, 2nd is desc_file.

    # Shuffle input file list if requested. Useful to ensure that we do not handle only the first X files, which are all from the same site.
    if shuffle_input_file_order:
        if verbose:
            print(f"Shuffling input file list with random seed '{random_seed}'.")
        random.seed(random_seed)
        random.shuffle(input_filepair_list)
    else:
        if verbose:
            print(f"Not shuffling input file list.")

    num_cores_tag = "all" if num_cores is None or num_cores == 0 else num_cores
    seq_par_tag = " sequentially " if sequential else f" in parallel using {num_cores_tag} cores"

    if verbose:
        print(f"Discovered {len(input_filepair_list)} valid pairs of input mesh and descriptor files.")

        if sequential:
            if num_neighborhoods_to_load is None:
                print(f"Will load all data from the {len(input_filepair_list)} files{seq_par_tag}.")
            else:
                print(f"Will load {num_neighborhoods_to_load} samples in total from the {len(input_filepair_list)} files.")
        else:
            if num_files_to_load is None:
                print(f"Will load data from all {len(input_filepair_list)} files{seq_par_tag}.")
            else:
                print(f"Will load data from {num_files_to_load} input files.")

        if num_samples_per_file is None:
            print(f"Will load all suitably sized vertex neighborhoods from each mesh file.")
        else:
            print(f"Will load at most {num_samples_per_file} vertex neighborhoods per mesh file.")

        neigh_size_tag = "auto-determined neighborhood size" if mesh_neighborhood_count is None or mesh_neighborhood_count == 0 else f"neighborhood size {mesh_neighborhood_count}"
        if filter_smaller_neighborhoods:
            print(f"Will filter (remove) all neighborhoods smaller than {neigh_size_tag}.")
        else:
            print(f"NOTICE: Will fill the respective missing columns of neighborhoods smaller than {neigh_size_tag} with NAN values. You will have to handle NAN values before training! (Set 'filter_smaller_neighborhoods' to 'True' to ignore them instead.)")


    load_start = time.time()
    tdl = TrainingData(neighborhood_radius=mesh_neighborhood_radius, num_neighbors=mesh_neighborhood_count)
    if sequential:
        dataset, col_names, datafiles_loaded = tdl.neighborhoods_from_raw_data_seq(input_filepair_list, num_samples_total=num_neighborhoods_to_load, num_samples_per_file=num_samples_per_file, add_desc_vertex_index=add_desc_vertex_index, add_desc_neigh_size=add_desc_neigh_size, filter_smaller_neighborhoods=filter_smaller_neighborhoods, exactly=exactly, add_desc_brain_bbox=add_desc_brain_bbox, add_subject_and_hemi_columns=add_subject_and_hemi_columns, random_seed=random_seed)
    else:
        dataset, col_names, datafiles_loaded = tdl.neighborhoods_from_raw_data_parallel(input_filepair_list, num_files_total=num_files_to_load, num_samples_per_file=num_samples_per_file, add_desc_vertex_index=add_desc_vertex_index, add_desc_neigh_size=add_desc_neigh_size, num_cores=num_cores, filter_smaller_neighborhoods=filter_smaller_neighborhoods, exactly=exactly, add_desc_brain_bbox=add_desc_brain_bbox, add_subject_and_hemi_columns=add_subject_and_hemi_columns, random_seed=random_seed)
    load_end = time.time()
    load_execution_time = load_end - load_start
    if verbose:
        print(f"=== Loading data files{seq_par_tag} done, it took: {timedelta(seconds=load_execution_time)} ===")

    data_settings['datafiles_loaded'] = datafiles_loaded

    assert isinstance(dataset, pd.DataFrame)
    return dataset, col_names, data_settings



def get_dataset_pickle(data_settings_in, do_pickle_data, dataset_pickle_file=None, dataset_settings_file=None, verbose=True):
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
        dataset, col_names, data_settings = compute_dataset(**data_settings_in)
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


