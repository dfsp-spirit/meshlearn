#!/usr/bin/env python


# This is just a temporary test script to test our data generator, ignore this.


# All credits for this go to Narasimha Karthik, this script is based on the one from his blog here:
# https://www.analyticsvidhya.com/blog/2022/02/approaching-regression-with-neural-networks-using-tensorflow/

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from sklearn.preprocessing import StandardScaler
import argparse

from meshlearn.data.generator import neighborhood_generator_reconall_dir

##### Settings #####

do_plot = True

##### End of settings #####

def train_lgi_dnn():
    """
    Train and evaluate an lGI prediction model using tensorflow neural network.
    """

    # Parse command line arguments
    example_text = '''Examples:

 # Use neighborhood radius 15mm, and coords/normals for max 300 vertices in radius, load 500k vertex neighborhoods:
 meshlearn_lgi_train -v -n 300 -r 15 -l 500000 -s -c 4 $SUBJECTS_DIR
 # Use neighborhood defaults, load 35k samples per file from 48 files, 8 in parallel. Persist dataset before training:
 meshlearn_lgi_train -p 35000 -f 48 -c 8 -t "_v2" -w . $SUBJECTS_DIR
 # After running the previous command, load the persisted dataset for training new model (with changed training settings in script):
 meshlearn_lgi_train -t "_v2" -w . $SUBJECTS_DIR
'''
    parser = argparse.ArgumentParser(prog='meshlearn_lgi_train_tf',
                                     description="Train and evaluate an DNN lGI prediction model.",
                                     epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter
                                     )

    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    parser.add_argument('data_dir', help="The recon-all data directory, created by FreeSurfer's recon-all on your sMRI images, or the directory containing the pickled data. Must be given unless -t is used and the input pkl file already exists.")
    parser.add_argument('-n', '--neigh_count', help="Number of vertices to consider at max in the edge neighborhoods for Euclidean dist.", default="100")
    parser.add_argument('-r', '--neigh_radius', help="Radius for sphere for Euclidean dist, in spatial units of mesh (e.g., mm).", default="10")
    parser.add_argument('-l', '--load_max', help="Total number of samples to load. Set to 0 for all in the files discovered in the data_dir. Used in sequential mode only.", default="0")
    parser.add_argument('-p', '--load_per_file', help="Total number of samples to load per file. Set to 0 for all in the respective mesh file. Useful to sample data from more different subjects and still not exhaust your RAM.", default="50000")
    parser.add_argument('-f', '--load_files', help="Total number of files to load. Set to 0 for all in the data_dir. Used in parallel mode only (see -s).", default="8")
    parser.add_argument("-s", "--sequential", help="Load data sequentially (as opposed to in parallel, the default). Not recommended. See also '-c'.", action="store_true")
    parser.add_argument("-c", "--cores", help="Number of cores to use when loading data in parallel. Defaults to all. (Model fitting always uses all cores.)", default=None)
    parser.add_argument("-t", "--pickle_tag", help="Optional, a tag (arbitrary string that will become a filename part) if you want to use pickling (saving/restoring) for datasets. If given the tag will be used to construct 1) the filename from/to which to unpickle/pickle the pre-processed dataset as 'ml<dataset_tag>_dataset.pkl', and 2) of the JSON metadata file for the dataset as 'ml<dataset_tag>_dataset.json'. If the model file does not exist, it will be created during the first run (with the respective JSON file), and used in subsequent runs with the same '--pickle-tag'. Can save a lot of time during model tuning if the dataset is final. Example: '_lgbmv1'.", default="")
    parser.add_argument("-w", "--write_dir", help="Optional writeable directory in which to save and from which to load pickled models and datasets, instead of in the data_dir. Useful if the latter is needed for the source data but is read-only. Ignored unless '-t' is also specified.")
    args = parser.parse_args()

    # Data settings not exposed on cmd line. Change here if needed.
    surface = 'pial'  # The mesh to use.
    descriptor = 'pial_lgi'  # The label descriptor, what you want to predict on the mesh.
    random_state = 42
    load_per_file_force_exactly = True # Whether to load exactly the requested number of entries per file, even if the file contains more (and more where thus read when reading it).

    # Preproc settings which are not exposed on the command line. (They are not exposed because changing them is most likely not need or a bad idea).
    cortex_label = False  # Whether to load FreeSurfer 'cortex.label' files and filter verts by them. Not implemented yet.
    add_desc_vertex_index = True  # whether to add vertex index as desriptor column to observation
    add_desc_neigh_size = True  # whether to add vertex neighborhood size (before pruning) as desriptor column to observation
    filter_smaller_neighborhoods = False  # Whether to filter (remove) neighborhoods smaller than 'args.neigh_count' (True), or fill the missing columns with 'np.nan' values instead. Note that, if you set to False, you will have to deal with the NAN values in some way before using the data, as most ML models cannot cope with NAN values.
    add_desc_brain_bbox = True
    add_local_mesh_descriptors = True
    add_global_mesh_descriptors = True

    ### Construct data settings from command line and other data setting above.

    ## All settings relevant for pre-processing of a single mesh. These must also be used when pre-processing meshes that you want to predict pvd-descriptors for later.
    preproc_settings = { 'cortex_label': cortex_label,
                        'add_desc_vertex_index':add_desc_vertex_index,
                        'add_desc_neigh_size':add_desc_neigh_size,
                        'mesh_neighborhood_radius':int(args.neigh_radius),
                        'mesh_neighborhood_count':int(args.neigh_count),
                        'filter_smaller_neighborhoods': filter_smaller_neighborhoods,
                        'add_desc_brain_bbox': add_desc_brain_bbox,
                        'add_local_mesh_descriptors' : add_local_mesh_descriptors,
                        'add_global_mesh_descriptors': add_global_mesh_descriptors
                    }

    ## All settings relevant for deciding which meshes to load, how to load them, and what data to keep from them.
    data_settings_in = {'data_dir': args.data_dir,
                        'num_neighborhoods_to_load': None if int(args.load_max) == 0 else int(args.load_max),
                        'surface': surface,
                        'descriptor' : descriptor,
                        'verbose': args.verbose,
                        'sequential': args.sequential,
                        'num_samples_per_file': None if int(args.load_per_file) == 0 else int(args.load_per_file),
                        'num_cores': None if (args.cores is None or args.cores == "0") else int(args.cores),
                        'num_files_to_load':None if int(args.load_files) == 0 else int(args.load_files),
                        'exactly': load_per_file_force_exactly
                        }


    ### Other settings, not related to data loading. Adapt here if needed.
    do_pickle_data = len(args.pickle_tag) > 0

    # Some common thing to identify a certain dataset. Freeform. Set to empty string if you do not need this.
    # Allows switching between pickled datasets quickly.
    dataset_tag = args.pickle_tag if args.pickle_tag is not None else ""
    model_tag = dataset_tag

    write_dir = args.data_dir if args.write_dir is None else args.write_dir

    dataset_pickle_file = os.path.join(write_dir, f"ml{dataset_tag}_dataset.pkl")  # Only relevant if do_pickle_data is True
    dataset_settings_file = os.path.join(write_dir, f"ml{dataset_tag}_dataset.json") # Only relevant if do_pickle_data is True
    training_history_image_filename = os.path.join(write_dir, f"ml{dataset_tag}_training.png")  # Image to save training history.

    do_persist_trained_model = True
    model_save_file = os.path.join(write_dir, f"ml{model_tag}_model.pkl")
    model_settings_file = os.path.join(write_dir, f"ml{model_tag}_model.json")



    ##### Define model #####

    # Now let's create a Deep Neural Network to train a regression model on our data.
    model = Sequential([
        layers.Dense(352, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer="RMSProp", loss="mean_absolute_error") #loss="mean_squared_error")

    ##### Fit model #####

    gen = neighborhood_generator_reconall_dir(batch_size=10000, data_settings=data_settings_in, preproc_settings=preproc_settings, verbose=True)
    batch = next(gen)
    #history = model.fit(epochs=25, x=train_features, y=train_labels, validation_data=(test_features, test_labels), verbose=1)

    ##### Analyze training #####

    def plot_loss(history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0,10])
        plt.xlabel('Epoch')
        plt.ylabel('Error (Loss)')
        plt.legend()
        plt.grid(True)

    #if do_plot:
    #    plt.ion()
    #    plot_loss(history)
    #    plt.show()


    # Model evaluation on testing dataset
    #model.evaluate(test_features, test_labels)

    model_output_file = "trained_meshlearn_model_edge_neigh_dist_5.h5"
    model.save(model_output_file)
    print(f"Saved trained model to '{model_output_file}'.")


    ### NOTE: To use saved model on new data:
    #saved_model = keras.models.load_model('trained_model.h5')
    #results = saved_model.predict(test_features)
    ## To look at results:
    #decoded_result = label_scaler.inverse_transform(results.reshape(-1,1))
    #print(decoded_result)

if __name__ == "__main__":
    train_lgi_dnn()


