"""
Read FreeSurfer brain meshes and pre-computed lgi per-vertex data for them from a directory.
"""

from genericpath import isdir, isfile
import brainload as bl
import trimesh as tm
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from meshlearn.neighborhood import neighborhoods_euclid_around_points, mesh_k_neighborhoods, mesh_neighborhoods_coords
from warnings import warn
import os.path
import glob

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

    def gen_data(self, datafiles, neighborhood_radius=25):
        """Generator for training data from files. Allows sample-wise loading of very large data sets.

        Use this or `load_data`, depending on whether or not you want everything in memory at once.

        Parameters
        ----------
        datafiles: dict str, str of mesh file names and corresponding per-vertex data file names. Must be FreeSurfer surf files and curv files.

        Yields
        ------
        X 2d nx3 float np.ndarray of neighborhood coordinates, each row contains the x,y,z coords of a single vertex. The n rows form the neighborhood around the source vertex.
        y scalar float, the per-vertex data value for the source vertex.
        """

        # TODO: this is not done yet

        for mesh_file_name, descriptor_file_name in datafiles.items():

            if not os.path.exists(mesh_file_name) and os.path.exists(descriptor_file_name):
                warn("Skipping non-existant file pair '{mf}' and '{df}'.".format(mf=mesh_file_name, df=descriptor_file_name))
                continue

            vert_coords, faces, pvd_data = TrainingData.data_from_files(mesh_file_name, descriptor_file_name)
            self.mesh = tm.Trimesh(vertices=vert_coords, faces=faces)

            if self.distance_measure == "Euclidean":
                self.kdtree = KDTree(vert_coords)
                neighborhoods = neighborhoods_euclid_around_points(vert_coords, self.kdtree, neighborhood_radius=neighborhood_radius)

            elif self.distance_measure == "graph":
                neighborhoods = mesh_k_neighborhoods(self.mesh, k=self.neighborhood_k)
                neighborhoods_centered_coords = mesh_neighborhoods_coords(neighborhoods, self.mesh, num_neighbors_max=self.num_neighbors)

            else:
                raise ValueError("Invalid distance_measure {dm}, must be one of 'graph' or 'Euclidean'.".format(dm=self.distance_measure))


            for vertex_idx in range(vert_coords.shape[0]):
                X =  neighborhoods_centered_coords[vertex_idx]
                y = pvd_data[vertex_idx]
                yield (X, y)




    def load_raw_data(self, datafiles, num_samples_to_load=None, neighborhood_radius=None, force_no_more_than_num_samples_to_load=False, df=True):
        """Loader for training data from FreeSurfer format (non-preprocessed) files, also does the preprocessing on the fly.

        Note that the data must fit into memory. Use this or `gen_data`, depending on whether or not you want everything in memory at once.

        Parameters
        ----------
        datafiles: dict str, str of mesh file names and corresponding per-vertex data file names. Must be FreeSurfer surf files and curv files.
        num_samples: positive integer, the number of samples to return from the files. Set to None to return all values. A sample consists of the data for a single vertex, i.e., its neighborhood coordinates and its target per-vertex value. Setting to None is slower, because we cannot pre-allocate.
        df : bool, whether to return as padnas.DataFrame (instead of numpy.ndarray)

        Returns
        ------
        X 2d nx3 float np.ndarray of neighborhood coordinates, each row contains the x,y,z coords of a single vertex. The n rows form the neighborhood around the source vertex.
        y scalar float, the per-vertex data value for the source vertex.
        """
        if neighborhood_radius is None:
            neighborhood_radius = self.neighborhood_radius

        if not isinstance(datafiles, dict):
            raise ValueError("datafiles must be a dict")
        if len(datafiles) == 0:
            raise ValueError("datafiles must not be empty")

        num_samples_loaded = 0
        do_break = False
        full_data = None
        for mesh_file_name, descriptor_file_name in datafiles.items():
            if do_break:
                break

            if not os.path.exists(mesh_file_name) and os.path.exists(descriptor_file_name):
                warn("Skipping non-existant file pair '{mf}' and '{df}'.".format(mf=mesh_file_name, df=descriptor_file_name))
                continue

            print(f"Loading mesh file '{mesh_file_name}' and descriptor file '{descriptor_file_name}'.")
            vert_coords, faces, pvd_data = TrainingData.data_from_files(mesh_file_name, descriptor_file_name)
            self.mesh = tm.Trimesh(vertices=vert_coords, faces=faces)

            if self.distance_measure == "Euclidean":
                self.kdtree = KDTree(vert_coords)
                print(f"Computing neighborhoods based on radius {neighborhood_radius} for {vert_coords.shape[0]} vertices in mesh file '{mesh_file_name}'.")
                neighborhoods, col_names = neighborhoods_euclid_around_points(vert_coords, self.kdtree, neighborhood_radius=neighborhood_radius, mesh=self.mesh, max_num_neighbors=self.num_neighbors, pvd_data=pvd_data)

                neighborhoods_size_bytes = getsizeof(neighborhoods)
                print(f"Current neighborhood size in RAM is about {neighborhoods_size_bytes} bytes, or {neighborhoods_size_bytes / 1024. / 1024.} MB.")

                if full_data is None:
                    full_data = neighborhoods
                else:
                    full_data = np.concatenate((full_data, neighborhoods,), axis=0)

                num_samples_loaded += neighborhoods.shape[0]
            else:
                raise ValueError("Invalid distance_measure {dm}, must be 'Euclidean'.".format(dm=self.distance_measure))



            if num_samples_to_load is not None:
                if num_samples_loaded >= num_samples_to_load:
                        print(f"Done loading the requested {num_samples_to_load} samples, ignoring the rest.")
                        do_break = True
                        break

        if num_samples_to_load is not None:
                if num_samples_loaded > num_samples_to_load:
                    if force_no_more_than_num_samples_to_load:
                        print(f"Truncating data of size {num_samples_loaded} to {num_samples_to_load} samples, 'force_no_more_than_num_samples_to_load' is true.")
                        full_data = full_data[0:num_samples_to_load, :] # this wastes stuff we spent time loading
                    else:
                        print(f"Returning {num_samples_loaded} instead of {num_samples_to_load} samples, file contained more and 'force_no_more_than_num_samples_to_load' is false.")


        if df:
            full_data = pd.DataFrame(full_data, columns=col_names)
            dataset_size_bytes = full_data.memory_usage(deep=True).sum()
            print(f"Total dataset size in RAM is about {dataset_size_bytes} bytes, or {dataset_size_bytes / 1024. / 1024.} MB.")


        return full_data



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
        raise ValueError("The data directory '{recon_dir}' does not exist or cannot be accessed".format(data_dir=recon_dir))

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
    valid_subjects = []

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
                        valid_subjects.append(subject)
                else:
                    if os.path.isfile(surf_file) and os.path.isfile(desc_file):
                        valid_mesh_files.append(surf_file)
                        valid_desc_files.append(desc_file)
                        valid_subjects.append(subject)


    if verbose:
        print(f"Out of {len(subjects_list)*2} subject hemispheres, {len(valid_mesh_files)} had the requested surface and descrpitor file.")

    return valid_mesh_files, valid_desc_files, valid_labl_files, valid_subjects





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






