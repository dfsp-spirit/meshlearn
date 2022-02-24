# Functions for loading data as a Tensorflow Dataset

# see the official TF data docs for an example with simulated data

# For an example with real-world data in files, see
# https://biswajitsahoo1111.github.io/post/efficiently-reading-multiple-files-in-tensorflow-2/



#from scipy.spatial import distance_matrix

import tensorflow as tf
import tensorflow_datasets as tfds
import nibabel.freesurfer.io as fsio

class VertexPropertyDataset(tf.data.Dataset):

    # Idea for organization of traning data: leave the meshes in FreeSurfer format files
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
    def _generator(datafiles, batch_size=20):
        # Opening the file
        time.sleep(0.03)
        mesh_file_list, descriptor_file_list = zip(*datafiles)

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)
            

            yield (sample_idx,)

    def _data_from_files(mesh_file_name, descriptor_file_name):
        vert_coords, faces = fsio.read_geometry(mesh_file_name)
        pvd_data = fsio.read_morph_data(descriptor_file_name)
        return(_transform_raw_data(vert_coords, faces, pvd_data))

    def _transform_raw_data(vertcoords, faces, pvd_data):


    def __new__(self, datafiles, batch_size=20):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature = tf.TensorSpec(shape = (2,), dtype = tf.float64),
            args=(datafiles, batch_size)
        )
