# Functions for loading data as a Tensorflow Dataset

# see the official TF data docs for an example with simulated data

# For an example with real-world data in files, see
# https://biswajitsahoo1111.github.io/post/efficiently-reading-multiple-files-in-tensorflow-2/



#from scipy.spatial import distance_matrix

from statistics import mean
import tensorflow as tf
import tensorflow_datasets as tfds
import nibabel.freesurfer.io as fsio
import trimesh as tm
import numpy as np

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

    # datafiles: dictionary<str,str>. the keys are mesh files, the values are the repective per-vertex descriptor files for the meshes.
    def _generator(self, datafiles, num_files=None):
        # Opening the file
        mesh_file_list, descriptor_file_list = zip(*datafiles)

        num_files_available = len(datafiles)
        if num_files is None:
            num_files = num_files_available
        else:
            num_files = min(num_files_available, num_files)

        for mesh_file_name, descriptor_file_name in datafiles.items():
            file_data = self._data_from_files(mesh_file_name, descriptor_file_name)
            # TODO: extract a single sample in loop here and yield it
            yield (sample_idx,)

    # Extract mesh and descriptor data from a single pair of files.
    def _data_from_files(self, mesh_file_name, descriptor_file_name):
        vert_coords, faces = fsio.read_geometry(mesh_file_name)
        pvd_data = fsio.read_morph_data(descriptor_file_name)
        return(_transform_raw_data(vert_coords, faces, pvd_data))

    # Compute the vertex neighborhood of the Tmesh for a given vertex
    def _transform_raw_data(self, vertcoords, faces, pvd_data):
        neighborhoods = []
        return neighborhoods


    def __new__(self, datafiles, num_files=None):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature = tf.TensorSpec(shape = (2,), dtype = tf.float64),
            args=(datafiles, num_files)
        )


# Compute the k-neighborhood for all vertices of a mesh.
# parameter tmesh must be a mesh instance from the trimesh package
def _k_neighborhoods(tmesh, k=1):
    neighborhoods = dict()
    print("Mesh has {nv} vertices, coords are in {d}d space.".format(nv=tmesh.vertices.shape[0], d=tmesh.vertices.shape[1]))
    print("Computing k-neighborhoods for k={step_idx}, will compute up to k={k}.".format(step_idx=1, k=k))
    for vert_idx in range(tmesh.vertices.shape[0]):
        neighborhoods[vert_idx] = np.array(tmesh.vertex_neighbors[vert_idx])
    if k == 1:
        return neighborhoods
    else:
        for step_idx in range(2, k+1):
            print("Computing k-neighborhoods for k={step_idx}, will compute up to k={k}.".format(step_idx=step_idx, k=k))
            for vert_idx in neighborhoods.keys():
                cur_neighbors = neighborhoods[vert_idx]
                neighborhoods[vert_idx] = np.unique(np.concatenate([neighborhoods.get(key) for key in cur_neighbors]))
    nsizes = np.array([len(v) for k,v in neighborhoods.items()])
    print("Neighborhood sizes are min={min}, max={max}, mean={mean}.".format(min=nsizes.min(), max=nsizes.max(), mean=nsizes.mean()))
    return neighborhoods


# Extract vertex coords of all neighborhood vertices and center them, so that
# the respective source vertex is at the origin.
def _neighborhoods_centered_coords(neighborhoods, tmesh, num_neighbors=10):
    neigh_coords = np.ndarray(shape=(num_neighbors, 3), dtype=float)
    vert_idx = 0
    for central_vertex, neighbors in neighborhoods.items():
        central_coords = tmesh.vertices[central_vertex, :]
        neigh_coords[vert_idx, :] = tmesh.vertices[central_vertex]




