#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import numpy as np
import tensorflow as tf
import argparse
import glob

from tensorflow.keras import layers
from tensorflow.keras import Sequential

from sklearn.model_selection import train_test_split
from meshlearn.tfdata import VertexPropertyDataset
from meshlearn.training_data import TrainingData

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# PYTHONPATH=./src/meshlearn python src/meshlearn/clients/meshlearn_lgi.py --verbose


def meshlearn_lgi():
    """
    Train and evaluate an lGI prediction model.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate an lGI prediction model.")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    parser.add_argument('-d', '--data-dir', help="The data directory. Use deepcopy_testdata.py script to create.", default="./tests/test_data/tim_only")
    parser.add_argument('-e', '--epochs', help="Number of training epochs.", default="20")
    parser.add_argument('-n', '--neigh_count', help="Number of vertices to consider at max in the edge neighborhoods for Euclidean dist.", default="50")
    parser.add_argument('-r', '--neigh_radius', help="Radius for sphere for Euclidean dist, in spatial units of mesh (e.g., mm).", default="10")
    args = parser.parse_args()

    num_epochs = int(args.epochs)
    data_dir = args.data_dir
    mesh_neighborhood_count = int(args.neigh_count) # How many vertices in the edge neighborhood do we consider (the 'local' neighbors from which we learn).
    mesh_neighborhood_radius = int(args.neigh_radius)

    print("---Train and evaluate an lGI prediction model---")
    if args.verbose:
        print("Verbosity turned on.")
        print("Training for {num_epochs} epochs.".format(num_epochs=num_epochs))
        print("Using data directory '{data_dir}'.".format(data_dir=data_dir))

    if not os.path.isdir(data_dir):
        raise ValueError("The data directory '{data_dir}' does not exist or cannot be accessed".format(data_dir=data_dir))

    mesh_files = np.sort(glob.glob("{data_dir}/*.pial".format(data_dir=data_dir)))
    descriptor_files = np.sort(glob.glob("{data_dir}/*.pial_lgi".format(data_dir=data_dir)))
    if args.verbose:
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

    if args.verbose:
        print("Found {num_valid_file_pairs} valid pairs of mesh file with matching descriptor file.".format(num_valid_file_pairs=num_valid_file_pairs))

    ### Decide which files are used as training, validation and test data. ###
    random_state = 42
    train_file_names, test_file_names = train_test_split(mesh_files, test_size = 0.2, random_state = random_state)
    train_file_names, validation_file_names = train_test_split(train_file_names, test_size = 0.1, random_state = random_state)

    train_file_dict = dict(zip(train_file_names, [k + "_lgi" for k in train_file_names]))
    validation_file_dict = dict(zip(validation_file_names, [k + "_lgi" for k in validation_file_names]))
    test_file_dict = dict(zip(test_file_names, [k + "_lgi" for k in test_file_names]))


    do_use_tfloader = False
    mesh_dim = 3                       # The number of mesh dimensions (x,y,z).
    num_neighborhoods_to_load = 50000
    batch_size = 32

    if do_use_tfloader:
        print(f"Using tensorflow loader.")
        # This is not functional yet.
        tf_data_generator = VertexPropertyDataset(train_file_dict)

        train_dataset = tf.data.Dataset.from_generator(tf_data_generator, args = [train_file_dict, batch_size],
                                                    output_shapes = ((None, mesh_neighborhood_count, mesh_dim, 1),(None,)),
                                                    output_types = (tf.float32, tf.float32))

        validation_dataset = tf.data.Dataset.from_generator(tf_data_generator, args = [validation_file_dict, batch_size],
                                                        output_shapes = ((None, mesh_neighborhood_count, mesh_dim, 1),(None,)),
                                                        output_types = (tf.float32, tf.float32))

        test_dataset = tf.data.Dataset.from_generator(tf_data_generator, args = [test_file_dict, batch_size],
                                                    output_shapes = ((None, mesh_neighborhood_count, mesh_dim, 1),(None,)),
                                                    output_types = (tf.float32, tf.float32))
    else:         # number of neighborhoods to load from training data. Set to None for all. Can limit here during development.
        print(f"Using basic loader to load {num_neighborhoods_to_load} samples.")
        tdl = TrainingData(distance_measure = "Euclidean", neighborhood_radius=mesh_neighborhood_radius, num_neighbors=mesh_neighborhood_count)
        train_dataset = tdl.load_data(train_file_dict, num_samples_to_load=num_neighborhoods_to_load, neighborhood_radius=mesh_neighborhood_radius)
        validation_dataset = tdl.load_data(validation_file_dict, num_samples_to_load=num_neighborhoods_to_load, neighborhood_radius=mesh_neighborhood_radius)
        test_dataset = tdl.load_data(test_file_dict, num_samples_to_load=num_neighborhoods_to_load, neighborhood_radius=mesh_neighborhood_radius)

    print(f"Loaded train, validation and test datasets with shapes: train={train_dataset.shape}, validation={validation_dataset.shape}, test{test_dataset.shape}.")

    # TODO: the shape of the training data is stupid. it currenlty has one vertex (x,y,z coords) per row. but it needs coords and normals of a neighborhood per row.

    ### Create the neural network model from layers ###
    #input_shape = (mesh_neighborhood_count, mesh_dim)
    print(f"=== Creating tensorflow model. ===")
    #model = tf.keras.Sequential([
    #    layers.Conv2D(16, 3, activation = "relu", input_shape = input_shape),
    #    layers.MaxPool2D(2),
    #    layers.Conv2D(32, 3, activation = "relu"),
    #    layers.MaxPool2D(2),
    #    layers.Flatten(),
    #    layers.Dense(16, activation = "relu"),
    #    layers.Dense(1, activation = "softmax")
    #])
    model = Sequential([
        layers.Dense(352, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    print(f"=== Training model. ===")
    steps_per_epoch = np.int(np.ceil(len(train_file_names)/batch_size))
    validation_steps = np.int(np.ceil(len(train_file_names)/batch_size))
    steps = np.int(np.ceil(len(test_file_names)/batch_size))
    print("steps_per_epoch = ", steps_per_epoch)
    print("validation_steps = ", validation_steps)
    print("steps = ", steps)

    model.fit(train_dataset, validation_data = validation_dataset, steps_per_epoch = steps_per_epoch, validation_steps = validation_steps, epochs = num_epochs)

    model.summary()

    test_loss, test_accuracy = model.evaluate(test_dataset, steps = steps)
    print("Test loss: ", test_loss)
    print("Test accuracy:", test_accuracy)

    sys.exit(0)


if __name__ == "__main__":
    meshlearn_lgi()
