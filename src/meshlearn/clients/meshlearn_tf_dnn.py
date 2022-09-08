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
from sklearn.preprocessing import StandardScaler

##### Settings #####

#data_dir = os.path.expanduser("~/develop/cpp_geodesics/output/")
data_dir = os.path.expanduser("/media/spirit/science/data/abide_meshlearn/")

max_files_to_load = 2  # Set to something <= 0 for unlimited.

##### End of settings #####


#### Functions #####

def load_data(data_dir, max_files_to_load):
    csv_files = glob.glob(os.path.join(data_dir, "*meshdist_edge_5_neigh.csv"))

    num_csv_files = len(csv_files)
    print(f"Discovered {num_csv_files} CSV files containing data in dir '{data_dir}'.")

    if max_files_to_load > 0:
        print(f"Will load up to {max_files_to_load} CSV files due to 'max_files_to_load' setting.")

    if num_csv_files < 1:
        raise ValueError(f"Found no CSV files with training data in directory '{data_dir}'.")

    dataset = pd.read_csv(csv_files[0], sep=" ")
    dataset_num_rows_with_na = dataset.isna().any(axis=0).sum()
    dataset_num_cols_with_na = dataset.isna().any(axis=1).sum()
    if dataset_num_rows_with_na > 0:
        print(f"  * Data loaded from file {csv_files[0]} contained {dataset_num_rows_with_na} rows with NA values (and {dataset_num_cols_with_na} columns). Dropping {dataset_num_rows_with_na} of {len(dataset)} rows.")
    dataset.dropna(axis=0, inplace=True)

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
                        print(f"  * Data loaded from file #{idx} contained {dset_num_rows_with_na} rows with NA values (and {dset_num_cols_with_na} columns). Dropping {dset_num_rows_with_na} of {len(dset)} rows.")
                    dset.dropna(axis=0, inplace=True)
                    dataset = pd.concat([dataset, dset], ignore_index=True)
                    num_files_loaded += 1
                else:
                    print(f" -At file #{idx}: Ignoring CSV file '{filename}': it has {len(dset_preview.columns)}, but existing dataset has {len(dataset.columns)}.")
                    num_incompatible += 1

    print(f"Loaded dataset with shape {dataset.shape} from {num_files_loaded} files. Ignored {num_incompatible} CSV files with incompatible column count.")
    return dataset


######################################## Start script ########################################


##### Load data #####

dataset = load_data(data_dir, max_files_to_load)

dataset_num_rows_with_na = dataset.isna().any(axis=0).sum()
dataset_num_cols_with_na = dataset.isna().any(axis=1).sum()

if dataset_num_rows_with_na > 0:
    print(f"Found {dataset_num_rows_with_na} rows with NA values in dataset (and {dataset_num_cols_with_na} columns). Dropping them.")
    dataset.dropna(axis=0, inplace=True)
else:
    print(f"Found no NA values in dataset.")


##### Split into train and test data #####


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Separate labels and features
label_column_name = "label"

train_features = train_dataset.drop([label_column_name], axis=1)
test_features = test_dataset.drop([label_column_name], axis=1)
train_labels = train_dataset[label_column_name]
test_labels = test_dataset[label_column_name]


##### Feature scaling #####

feature_scaler = StandardScaler()
label_scaler = StandardScaler()

##  Fit on Training Data
feature_scaler.fit(train_features.values)
label_scaler.fit(train_labels.values.reshape(-1, 1))

## Transform both training and testing data
train_features = feature_scaler.transform(train_features.values)
test_features = feature_scaler.transform(test_features.values)
train_labels = label_scaler.transform(train_labels.values.reshape(-1, 1))
test_labels = label_scaler.transform(test_labels.values.reshape(-1, 1))


##### Define model #####

# Now let's create a Deep Neural Network to train a regression model on our data.
model = Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer="RMSProp",
              loss="mean_squared_error")

##### Fit model #####

history = model.fit(epochs=100, x=train_features, y=train_labels,
          validation_data=(test_features, test_labels), verbose=1)

##### Analyze training #####

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,10])
    plt.xlabel('Epoch')
    plt.ylabel('Error (Loss)')
    plt.legend()
    plt.grid(True)

plot_loss(history)


