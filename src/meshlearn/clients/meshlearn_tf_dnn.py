#!/usr/bin/env python


# All credits for this go to Narasimha Karthik, this script is based on the one from his blog here:
# https://www.analyticsvidhya.com/blog/2022/02/approaching-regression-with-neural-networks-using-tensorflow/

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

from meshlearn.load_preproc_data import load_preproc_csv_data

##### Settings #####

#data_dir = os.path.expanduser("~/develop/cpp_geodesics/output/")
data_dir = os.path.expanduser("/media/spirit/science/data/abide_meshlearn/")

max_files_to_load = 10  # Set to something <= 0 for unlimited.

do_plot = True

##### End of settings #####



######################################## Start script ########################################


##### Load data #####

dataset = load_preproc_csv_data(data_dir, max_files_to_load)

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
    layers.Dense(352, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer="RMSProp", loss="mean_absolute_error") #loss="mean_squared_error")

##### Fit model #####

history = model.fit(epochs=25, x=train_features, y=train_labels,
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

if do_plot:
    plt.ion()
    plot_loss(history)
    plt.show()


# Model evaluation on testing dataset
model.evaluate(test_features, test_labels)

model_output_file = "trained_meshlearn_model_edge_neigh_dist_5.h5"
model.save(model_output_file)
print(f"Saved trained model to '{model_output_file}'.")


### NOTE: To use saved model on new data:
#saved_model = models.load_model('trained_model.h5')
#results = saved_model.predict(test_features)
## To look at results:
#decoded_result = label_scaler.inverse_transform(results.reshape(-1,1))
#print(decoded_result)


