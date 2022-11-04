[![status](https://github.com/dfsp-spirit/meshlearn/actions/workflows/cov_test_workflow.yml/badge.svg)](https://github.com/dfsp-spirit/meshlearn/actions)

# meshlearn
AI model to predict computationally expensive local, vertex-wise descriptors like the local gyrification index from the local mesh neighborhood.

This includes a python package and API (`meshlearn`) and two command line applications for training and predicting lGI, `meshlearn_lgi_train` and `meshlearn_lgi_predict`. End users are most likely interested only in the `meshlearn_lgi_predict` command, in combination with one of our pre-trained models.

![Vis0](./web/meshlearn_brain_blank_and_lgi_web.jpg?raw=true "Left: Brain surface, faces drawn. Right: Visualization of predicted lGI per-vertex data on the mesh, using the viridis colormap.")
**Fig. 0** *Left: Brain surface, faces drawn. Right: Visualization of predicted lGI per-vertex data on the mesh, using the viridis colormap.*


## About

Predict per-vertex descriptors like the local gyrification index (lGI) or other local descriptors for a mesh.

* The local gyrification index is a brain morphometry descriptor used in computational neuroimaging. It describes the folding of the human cortex at a specific point, based on a mesh reconstruction of the cortical surface from a magnetic resonance image (MRI). See [Schaer et al. 2008](https://doi.org/10.1109/TMI.2007.903576) for details.
* The geodesic circle radius and related descriptors are described in my [cpp_geodesics repo](https://github.com/dfsp-spirit/cpp_geodesics) and in the references listed there. Ignore the global descriptors (like mean geodesic distance) in there.


![Vis1](./web/brain_mesh_full.jpg?raw=true "Brain mesh, white surface.")

**Fig. 1** *A mesh representing the human cortex, edges drawn.*

![Vis2](./web/brain_mesh_vertices.jpg?raw=true "Brain mesh, zoomed view that shows the mesh structure.")

**Fig. 2** *Close up view of the triangular mesh, showing the vertices, edges and faces. Each vertex neighborhood (example for the ML model) describes the mesh structure in a sphere around the respective vertex. Vertex neighborhoods are computed from the mesh during pre-processing.*

This implementation uses Python, with `tensorflow` and `lightgbm` for the machine learning part. Mesh pre-processing is done with `pymesh` and `igl`.

## Why

Computing lGI and some other mesh properties for brain surface meshes is slow and sometimes fails even for good quality meshes, leading to exclusion of the respective MRI scans. The lGI computation also requires Matlab, which is inconvenient and prevents the computation of lGI on high performance computer clusters (due to the excessive licensing costs), which would be a way to deal with the long computation times. This project aims to provide a trained model that will predict the lGI for a vertex based on the mesh neighborhood. The aim is to have a faster and more robust method to compute lGI, based on free software.

## Usage

### Predicting using pre-trained models

**Please keep in mind that meshlearn is in the alpha stage, use in production is not yet recommended. You are free to play around with it though!**

Currently meshlearn comes with one pre-trained model for predicting the local gyrification index (lGI, Schaer et al.) for full-resolution, native space [FreeSurfer meshes](https://freesurfer.net/). These meshes are (a part of) the result of running FreeSurfer's `recon-all` pipeline on structural MRI scans of the human brain.

The model is a gradiant-boosting machine as implemented in [lightgbm](https://github.com/microsoft/LightGBM), and it was trained on a diverse training set of about 60 GB of pre-processed mesh data, obtained from the publicly available, multi-site [ABIDE I dataset](https://fcon_1000.projects.nitrc.org/indi/abide/). The model can be found at [tests/test_data/models/lgbm_lgi/](./tests/test_data/models/lgbm_lgi/), and consists of the model file (`ml_model.pkl`, the pickled lightgbm model) and a metadata file ([ml_model.json](tests/test_data/models/lgbm_lgi/ml_model.json)) that contains the pre-processing settings used to train the model. These settings must also be used when predicting for a new mesh.

The `meshlearn_lgi_predict` command line application that is part of meshlearn can be used to predict lGI for your own FreeSurfer meshes using the supplied model or alternative models. After installation of meshlearn, run `meshlearn_lgi_predict --help` for available options. (For now, you will need to follow the installation instructions in the development section below, as there is not official release yet.)

Information on model performance can be found in the mentioned [ml_model.json file](tests/test_data/models/lgbm_lgi/ml_model.json), under the key `model_info.evaluation`. The model has not been fine-tuned yet.

### Training your own model

If you want to *train your own model* instead of using one of our models, you will need suitable training data, Matlab and a powerful multi-core machine with 128+ GB of RAM. Please see the [development instructions](./README_DEV.md) for more details.

