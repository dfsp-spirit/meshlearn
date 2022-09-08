#!/usr/bin/env python
from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers


##### Settings #####

#data_dir = os.path.expanduser("~/develop/cpp_geodesics/output/")
data_dir = os.path.expanduser("/media/spirit/science/data/abide_meshlearn/")

max_files_to_load = 8  # Set to something <= 0 for unlimited.

##### End of settings #####


def load_data(data_dir, max_files_to_load):
    csv_files = glob.glob(os.path.join(data_dir, "*meshdist_edge_5_neigh.csv"))

    num_csv_files = len(csv_files)
    print(f"Discovered {num_csv_files} CSV files containing data in dir '{data_dir}'.")

    if max_files_to_load > 0:
        print(f"Will load up to {max_files_to_load} CSV files due to 'max_files_to_load' setting.")

    if num_csv_files < 1:
        raise ValueError(f"Found no CSV files with training data in directory '{data_dir}'.")

    dataset = pd.read_csv(csv_files[0], sep=" ")

    print(f"Loaded initial dataset with shape {dataset.shape}.")

    ## Read additional CSV files and merge results with 1st one. This may flood RAM for many
    ## large CSV files on less powerful machines. In that case, you will have to use `partial_fit` and train on chunks.
    num_files_loaded = 1
    num_incompatible = 0
    if num_csv_files > 1:
        print(f"Loading additional datasets from the {num_csv_files} files.")
        for idx, filename in enumerate(csv_files):
            if idx == 0:
                print(f" -At file #{idx}: '{filename}': Skipping first one, already loaded")
                continue  # The first one was already loaded.

            if max_files_to_load > 0 and num_files_loaded >= max_files_to_load:   # Limit
                print(f"Loaded {num_files_loaded} CSV files with a total of {len(dataset)} rows so far, ignoring the rest due to 'max_files_to_load'={max_files_to_load} setting.")
                break
            else:   # No limit, keep loading.
                dset_preview = pd.read_csv(filename, sep=" ", nrows=1)
                if len(dset_preview.columns) == len(dataset.columns):
                    dset = pd.read_csv(filename, sep=" ")
                    print(f" -At file #{idx}: Adding {len(dset)} rows from CSV file '{filename}' to current dataset with {len(dataset)} rows.")
                    dset_num_rows_with_na = dset.isna().any(axis=0).sum()
                    dset_num_cols_with_na = dset.isna().any(axis=1).sum()
                    if dset_num_rows_with_na > 0:
                        print(f"  * Data loaded from file #{idx} contained {dset_num_rows_with_na} rows with NA values (and {dset_num_cols_with_na} columns). Dropping {dset_num_rows_with_na} of {len(dataset)} rows.")
                    dset.dropna(axis=0, inplace=True)
                    dataset = pd.concat([dataset, dset], ignore_index=True)
                    num_files_loaded += 1
                else:
                    print(f" -At file #{idx}: Ignoring CSV file '{filename}': it has {len(dset_preview.columns)}, but existing dataset has {len(dataset.columns)}.")
                    num_incompatible += 1

    print(f"Loaded dataset with shape {dataset.shape} from {num_files_loaded} files. Ignored {num_incompatible} CSV files with incompatible column count.")
    return dataset

dataset = load_data(data_dir, max_files_to_load)

dataset_num_rows_with_na = dataset.isna().any(axis=0).sum()
dataset_num_cols_with_na = dataset.isna().any(axis=1).sum()

if dataset_num_rows_with_na > 0:
    print(f"WARNING: Found {dataset_num_rows_with_na} rows with NA values in dataset (and {dataset_num_cols_with_na} columns).")
else:
    print(f"Found no NA values in dataset")




