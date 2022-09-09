"""
Read FreeSurfer brain meshes and pre-computed lgi per-vertex data for them from a directory.
"""

import brainload as bl
import trimesh as tm
import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.spatial import KDTree
from meshlearn.neighborhood import neighborhoods_euclid_around_points, mesh_k_neighborhoods, mesh_neighborhoods_coords
from warnings import warn
import os.path


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

    def load_data(self, datafiles, num_samples_to_load=None, neighborhood_radius=None):
        """Loader for training data from files.

        Note that the data must fit into memory. Use this or `gen_data`, depending on whether or not you want everything in memory at once.

        Parameters
        ----------
        datafiles: dict str, str of mesh file names and corresponding per-vertex data file names. Must be FreeSurfer surf files and curv files.
        num_samples: positive integer, the number of samples to return from the files. Set to None to return all values. A sample consists of the data for a single vertex, i.e., its neighborhood coordinates and its target per-vertex value. Setting to None is slower, because we cannot pre-allocate.

        Returns
        ------
        X 2d nx3 float np.ndarray of neighborhood coordinates, each row contains the x,y,z coords of a single vertex. The n rows form the neighborhood around the source vertex.
        y scalar float, the per-vertex data value for the source vertex.
        """
        if neighborhood_radius is None:
            self.neighborhood_radius = self.neighborhood_radius


        if not num_samples_to_load is None:
            full_data = np.empty((num_samples_to_load * self.num_neighbors, 3,), dtype=np.float)
            print(f"Will load up to {num_samples_to_load} samples from the {len(datafiles)} input file pairs.")
        else:
            full_data = np.empty((0, 3), dtype=np.float) # Will be expanded as needed, which is slow.
            print(f"Will load all data from the {len(datafiles)} input file pairs.")

        print(f"Will use distance metric {self.distance_measure}")

        num_samples_loaded = 0
        do_break = False
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
                print(f"Computing neighborhoods based on Euklidean radius {neighborhood_radius} for '{vert_coords.shape[0]}' vertices in mesh file '{mesh_file_name}'.")
                neighborhoods = neighborhoods_euclid_around_points(vert_coords, self.kdtree, neighborhood_radius=neighborhood_radius)
                neighborhoods = { i : neighborhoods[i] for i in range(0, len(neighborhoods) ) } # Convert list to dict.

                neighborhoods_centered_coords = mesh_neighborhoods_coords(neighborhoods, self.mesh, num_neighbors_max=self.num_neighbors)

            elif self.distance_measure == "graph":
                print(f"Computing neighborhoods based on graph edge distance {self.neighborhood_k} for '{vert_coords.shape[0]}' vertices in mesh file '{mesh_file_name}'.")
                neighborhoods = mesh_k_neighborhoods(self.mesh, k=self.neighborhood_k)
                neighborhoods_centered_coords = mesh_neighborhoods_coords(neighborhoods, self.mesh, num_neighbors_max=self.num_neighbors)

            else:
                raise ValueError("Invalid distance_measure {dm}, must be one of 'graph' or 'Euclidean'.".format(dm=self.distance_measure))


            for vertex_idx in range(vert_coords.shape[0]):
                if num_samples_to_load is not None:
                    if num_samples_loaded >= num_samples_to_load:
                        print(f"Done loading the requested {num_samples_to_load} samples, ignoring the rest.")
                        do_break = True
                        break

                neighborhood_start_idx = num_samples_loaded * self.num_neighbors
                neighborhood_end_idx = neighborhood_start_idx + self.num_neighbors
                full_data[neighborhood_start_idx:neighborhood_end_idx, :] = neighborhoods_centered_coords[vertex_idx]
                y = pvd_data[vertex_idx]
                num_samples_loaded += 1

        return full_data[0:(num_samples_loaded+1)*self.num_neighbors,:]






