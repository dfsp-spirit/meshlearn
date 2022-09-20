"""
Read FreeSurfer brain meshes and pre-computed lgi per-vertex data for them from a directory.
"""

import brainload as bl
import trimesh as tm
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from meshlearn import neighborhoods_euclid_around_points
from warnings import warn
import os.path
import glob

import psutil
from sys import getsizeof

def load_piallgi_morph_data(subjects_dir, subjects_list):
    return bl.group_native("pial_lgi", subjects_dir, subjects_list)


def load_surfaces(subjects_dir, subjects_list, surf="pial"):
    meshes = {}
    for subject in subjects_list:
        vertices, faces, meta_data = bl.subject_mesh(subject, subjects_dir, surf=surf)
        #meshes[subject] = { "vertices": vertices, "faces" : faces }
        meshes[subject] = tm.Trimesh(vertices=vertices, faces=faces)
    return meshes


class TrainingData():

    def __init__(self, distance_measure = "Euclidean", neighborhood_radius=20.0, neighborhood_k=2, num_neighbors=10, allow_nan=False):
        """
        Parameters
        ----------
        datafiles: dict str,str of mesh_file_name : pvd_file_name
        num_files: the number of file pairs to use (from the number available in datafiles). Set to None to use all.
        distance_measure: one of "Euclidean" or "graph"
        neighborhood_radius: only used with distance_measure = Euclidean. The ball radius for kdtree ball queries, defining the neighborhood.
        neighborhood_k: only used with distance_measure = graph. The k for the k-neighborhood in the mesh, i.e., the hop distance along edges in the graph that defines the neighborhood.
        num_neighbors: int, used with both distance_measures: the number of neighbors to actually use per vertex (keeps only num_neighbors per neighborhood), to fix the dimension of the descriptor. Each vertex must have at least this many neighbors for this to work, unless allow_nan is True.
        allow_nan: boolean, whether to continue in case a neighborhood is smaller than num_neighbors. If this is True and the situation occurs, np.nan values will be added as neighbor coordinates. If this is False and the situation occurs, an error is raised. Whether or not allowing nans makes sense depends on the machine learning method used downstream. Many methods cannot handle nan values.
        """
        self.distance_measure = distance_measure
        self.neighborhood_radius = neighborhood_radius
        self.neighborhood_k = neighborhood_k
        self.num_neighbors = num_neighbors
        self.allow_nan = allow_nan

        self.kdtree = None
        self.mesh = None
        self.distance_measure = distance_measure

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
        return (vert_coords, faces, pvd_data)

    # def gen_data(self, datafiles, neighborhood_radius=25):
    #     """Generator for training data from files. Allows sample-wise loading of very large data sets.
    #     Use this or `load_data`, depending on whether or not you want everything in memory at once
    #     Parameters
    #     ----------
    #     datafiles: dict str, str of mesh file names and corresponding per-vertex data file names. Must be FreeSurfer surf files and curv files.
    #     Yields
    #     ------
    #     X 2d nx3 float np.ndarray of neighborhood coordinates, each row contains the x,y,z coords of a single vertex. The n rows form the neighborhood around the source vertex.
    #     y scalar float, the per-vertex data value for the source vertex.
    #     """
    #     # TODO: this is not done yet
    #     for mesh_file_name, descriptor_file_name in datafiles.items():
    #         if not os.path.exists(mesh_file_name) and os.path.exists(descriptor_file_name):
    #             warn("Skipping non-existant file pair '{mf}' and '{df}'.".format(mf=mesh_file_name, df=descriptor_file_name))
    #             continue
    #         vert_coords, faces, pvd_data = TrainingData.data_from_files(mesh_file_name, descriptor_file_name)
    #         self.mesh = tm.Trimesh(vertices=vert_coords, faces=faces)
    #         if self.distance_measure == "Euclidean":
    #             self.kdtree = KDTree(vert_coords)
    #             neighborhoods = neighborhoods_euclid_around_points(vert_coords, self.kdtree, neighborhood_radius=neighborhood_radius)
    #         elif self.distance_measure == "graph":
    #             neighborhoods = mesh_k_neighborhoods(self.mesh, k=self.neighborhood_k)
    #             neighborhoods_centered_coords = mesh_neighborhoods_coords(neighborhoods, self.mesh, num_neighbors_max=self.num_neighbors)
    #         else:
    #             raise ValueError("Invalid distance_measure {dm}, must be one of 'graph' or 'Euclidean'.".format(dm=self.distance_measure))
    #         for vertex_idx in range(vert_coords.shape[0]):
    #             X =  neighborhoods_centered_coords[vertex_idx]
    #             y = pvd_data[vertex_idx]
    #             yield (X, y)


    def neighborhoods_from_raw_data_parallel(self, datafiles, neighborhood_radius=None, exactly=False, num_samples_per_file=None, df=True, verbose=False, max_num_neighbors=None, add_desc_vertex_index=False, add_desc_neigh_size=False, num_cores=8, num_files_total=None):
        """
        Parallel version of `neighborhoods_from_raw_data`.

        Note: Parameter 'num_samples_total' is not supported in parallel mode.
              Use 'num_files_total' (and 'num_samples_per_file') instead
              to limit total number of entries."
        """
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        if num_files_total is not None:
            if num_files_total < len(datafiles):
                keys_to_extract = list(datafiles)[0:num_files_total]
                datafiles_subset = {k: datafiles[k] for k in keys_to_extract}
                assert len(datafiles_subset) == num_files_total
                datafiles = datafiles_subset

        with ThreadPoolExecutor(num_cores) as pool:
            neighborhoods_from_raw_single_file_pair = partial(self.neighborhoods_from_raw_data, self=self, neighborhood_radius=neighborhood_radius, num_samples_total=None, exactly=exactly, num_samples_per_file=num_samples_per_file, df=df, verbose=verbose, max_num_neighbors=max_num_neighbors, add_desc_vertex_index=add_desc_vertex_index, add_desc_neigh_size=add_desc_neigh_size)
            df = pd.concat(pool.map(neighborhoods_from_raw_single_file_pair, datafiles))
        return df


    def neighborhoods_from_raw_data(self, datafiles, neighborhood_radius=None, num_samples_total=None, exactly=False, num_samples_per_file=None, df=True, verbose=True, max_num_neighbors=None, add_desc_vertex_index=False, add_desc_neigh_size=False):
        """Loader for training data from FreeSurfer format (non-preprocessed) files, also does the preprocessing on the fly.

        Will load mesh and descriptor files, and use a kdtree to quickly find, for each vertex, all neighbors withing Euclidean distance 'neighborhood_radius'.
        Returns, for each such vertex neighborhood, the coordinates and normals of all neighborhood vertices.
        Note that the data must fit into memory. Use this or `gen_data`, depending on whether or not you want everything in memory at once.

        Parameters
        ----------
        datafiles: dict str, str of mesh file names and corresponding per-vertex data file names. Must be FreeSurfer surf files and curv files.
        neighborhood_radius: radius for neighborhood sphere, in mesh units (mm for FreeSurfer meshes)
        num_samples_total: positive integer, the total number of samples (neighborhoods) to return from the mesh files. Set to None to return all values. A sample consists of the data for a single vertex, i.e., its neighborhood coordinates and its target per-vertex value. Setting to None is slower, because we cannot pre-allocate.
        exactly: bool, whether to force loading exactly 'num_samples_total' samples. If false, and the last chunk loaded from a file leads to more samples, this function will return all loaded ones. If true, the extra ones will be discarded and exactly 'num_samples_total' samples will be returned.
        num_samples_per_file: positive integer, the number of samples (neighborhoods) to load at max per mesh file. Can be used to read data from more different subjects, while still keeping the total training data size reasonable. Note that the function may return less, if filtering by size is active via `max_num_neighbors`.
        df : bool, whether to return as pandas.DataFrame (instead of numpy.ndarray)
        verbose: bool, whether to print output (or be silent)

        Returns
        ------
        X 2d nx3 float np.ndarray of neighborhood coordinates, each row contains the x,y,z coords of a single vertex. The n rows form the neighborhood around the source vertex.
        y scalar float, the per-vertex data value for the source vertex.
        """
        if neighborhood_radius is None:
            neighborhood_radius = self.neighborhood_radius

        if max_num_neighbors is None:
            max_num_neighbors = self.num_neighbors

        if not isinstance(datafiles, dict):
            raise ValueError("datafiles must be a dict")
        if len(datafiles) == 0:
            raise ValueError("datafiles must not be empty")

        num_samples_loaded = 0
        do_break = False
        full_data = None

        num_files_loaded = 0
        if verbose:
                print(f"[load] Loading data.")
        for mesh_file_name, descriptor_file_name in datafiles.items():
            if do_break:
                break

            if not os.path.exists(mesh_file_name) and os.path.exists(descriptor_file_name):
                warn("[load] Skipping non-existant file pair '{mf}' and '{df}'.".format(mf=mesh_file_name, df=descriptor_file_name))
                continue

            if verbose:
                print(f"[load] * Loading mesh file '{mesh_file_name}' and descriptor file '{descriptor_file_name}'.")
            vert_coords, faces, pvd_data = TrainingData.data_from_files(mesh_file_name, descriptor_file_name)
            self.mesh = tm.Trimesh(vertices=vert_coords, faces=faces)

            num_verts_total = vert_coords.shape[0]

            if num_samples_per_file == None:
                query_vert_coords = vert_coords
                query_vert_indices = np.arange(num_verts_total)
            else:
                query_vert_coords = vert_coords.copy()
                # Sample 'num_samples_per_file' vertex coords from the full coords list
                randomstate = np.random.default_rng(0)
                query_vert_indices = randomstate.choice(num_verts_total, num_samples_per_file, replace=False, shuffle=False)
                query_vert_coords = query_vert_coords[query_vert_indices, :]

            if self.distance_measure == "Euclidean":
                self.kdtree = KDTree(vert_coords)
                if verbose:
                    print(f"[load]  - Computing neighborhoods based on radius {neighborhood_radius} for {query_vert_coords.shape[0]} of {num_verts_total} vertices in mesh file '{mesh_file_name}'.")
                neighborhoods, col_names, kept_vertex_indices_mesh = neighborhoods_euclid_around_points(query_vert_coords, query_vert_indices, self.kdtree, neighborhood_radius=neighborhood_radius, mesh=self.mesh, max_num_neighbors=max_num_neighbors, pvd_data=pvd_data, add_desc_vertex_index=add_desc_vertex_index, add_desc_neigh_size=add_desc_neigh_size, verbose=verbose)

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
            else:
                raise ValueError("Invalid distance_measure {dm}, must be 'Euclidean'.".format(dm=self.distance_measure))



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
            dataset_size_bytes = full_data.memory_usage(deep=True).sum()
            if verbose:
                print(f"[load] Total dataset size in RAM is about {int(dataset_size_bytes / 1024. / 1024.)} MB.")
                print(f"[load] RAM available is about {int(psutil.virtual_memory().available / 1024. / 1024.)} MB")


        return full_data, col_names



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
    hemis list of str, containing one or more of 'lh', 'rh'
    cortex_label bool, whether to also require label/<hemi>.cortex.label files.
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

    valid_mesh_files = []
    valid_desc_files = []
    valid_labl_files = []
    subjects_valid = []
    subjects_missing_some_file = []

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
                        subjects_valid.append(subject)
                    else:
                        subjects_missing_some_file.append(subject)
                else:
                    if os.path.isfile(surf_file) and os.path.isfile(desc_file):
                        valid_mesh_files.append(surf_file)
                        valid_desc_files.append(desc_file)
                        subjects_valid.append(subject)
                    else:
                        subjects_missing_some_file.append(subject)

    if verbose:
        print(f"Out of {len(subjects_list)*2} subject hemispheres ({len(subjects_list)} subjects), {len(valid_mesh_files)} had the requested surface and descriptor files.")
        if len(subjects_missing_some_file) > 0:
            print(f"The following {len(subjects_missing_some_file)} subjects where missing files: {', '.join(subjects_missing_some_file)}")

    return valid_mesh_files, valid_desc_files, valid_labl_files, subjects_valid, subjects_missing_some_file





def get_valid_mesh_desc_lgi_file_pairs(dc_data_dir, verbose=True):
        """
        Discover valid pairs of mesh and descriptor files in datadir created with `deepcopy_testdata.py` script.

        WARNING: Note that `dc_data_dir` is NOT a standard FreeSurfer directory structure, but a flat directory with
                renamed files (including subject to make them unique in the dir). Use the mentioned script `deepcopy_testdata.py` to
                turn a FreeSUrfer recon-all output dir into such a flat dir.

        TODO: We should maybe rewrite this function to just work directly on a recon-all output dir.

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






