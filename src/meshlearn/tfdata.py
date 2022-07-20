# Functions for loading data as a Tensorflow Dataset

# see the official TF data docs for an example with simulated data

# For an example with real-world data in files, see
# https://biswajitsahoo1111.github.io/post/efficiently-reading-multiple-files-in-tensorflow-2/



#from scipy.spatial import distance_matrix

from statistics import mean

import tensorflow as tf
import nibabel.freesurfer.io as fsio
import trimesh as tm
from scipy.spatial import KDTree
from meshlearn.neighborhood import neighborhoods_euclid_around_points, mesh_k_neighborhoods, mesh_neighborhoods_coords

class VertexPropertyDataset(tf.data.Dataset):

    # Idea for organization of training data: leave the meshes in FreeSurfer format files
    # and read the coordinates of the *n_neigh* closest vertices (along mesh edges) from the files for each vertex (read using nibabel or trimesh, based
    # on the file extension maybe).
    # The mesh k-ring neighborhood needs to be computed for this (with a k-value that yields at least n vertices),
    # which can be done with trimesh. Then compute the distance matrix for the neighborhood vertices and take the BN closest ones, and
    # get their coordinates. This local neighborhood (given by coordinates relative to the current vertex) represents the mesh structure
    # that we should learn the descriptor value from.
    # The descriptor value itself can be parsed from FreeSurfer curv files, or optionally from CSV files. Maybe we should support both.
    #
    # Thus, this data generator needs to open 2 separate files to obtain a full training data set: the mesh file and the mesh descriptor file. This
    # also means that when we train to predict different mesh descriptors for a mesh, we do not need to store the same mesh several times on disk.
    """
    Parameters
    ----------
    datafiles: dictionary<str,str>. the keys are mesh files, the values are the repective per-vertex descriptor files for the meshes.
    num_files: positive integer, the number of files to use. For the default `None`, all files are used.
    neighborhood_radius: float, radius around point to use for neighborhood definition based on Euclidean distance.
    """

    kdtree = None
    mesh = None
    distance_measure = "Euclidean"


    def _generator(self, datafiles):
        # Opening the file
        #mesh_file_list, descriptor_file_list = zip(*datafiles)

        num_files_available = len(datafiles)
        if self.num_files is None:
            self.num_files = num_files_available
        else:
            if self.num_files > num_files_available:
                print("Requested {num_req} data files, but only {num_avail} available. Will use all available ones.".format(num_req=self.num_files, num_avail=num_files_available))
            self.num_files = min(num_files_available, self.num_files)

        num_files_handled = 0
        for mesh_file_name, descriptor_file_name in datafiles.items():
            while num_files_handled < self.num_files:
                vert_coords, faces, pvd_data = self._data_from_files(mesh_file_name, descriptor_file_name)

                if self.distance_measure == "Euclidean":
                    self.kdtree = KDTree(vert_coords)
                    neighborhoods = neighborhoods_euclid_around_points(vert_coords, self.kdtree)

                elif self.distance_measure == "graph":
                    self.mesh = tm.Trimesh(vertices=vert_coords, faces=faces)
                    neighborhoods = mesh_k_neighborhoods(self.mesh, k=self.neighborhood_k)
                    neighborhoods_centered_coords = mesh_neighborhoods_coords(neighborhoods, self.mesh, num_neighbors=self.num_neighbors)

                else:
                    raise ValueError("Invalid distance_measure {dm}, must be one of 'graph' or 'Euclidean'.".format(dm=self.distance_measure))


                for vertex_idx in range(vert_coords.shape[0]):
                    X =  neighborhoods_centered_coords[vertex_idx]
                    y = pvd_data[vertex_idx]
                    yield (X, y)



    # Extract mesh and descriptor data from a single pair of files.
    def _data_from_files(self, mesh_file_name, descriptor_file_name):
        vert_coords, faces = fsio.read_geometry(mesh_file_name)
        pvd_data = fsio.read_morph_data(descriptor_file_name)
        return (vert_coords, faces, pvd_data)


    def __new__(self, datafiles, num_files=None, distance_measure = "Euclidean", neighborhood_radius=20.0, neighborhood_k=2, num_neighbors=10, allow_nan=False):
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
        self.num_files = num_files
        self.distance_measure = distance_measure
        self.neighborhood_radius = neighborhood_radius
        self.neighborhood_k = neighborhood_k
        self.num_neighbors = num_neighbors
        self.allow_nan = allow_nan
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature = tf.TensorSpec(shape = (2,), dtype = tf.float64),
            args=(datafiles)
        )


