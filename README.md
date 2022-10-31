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

**Please keep in mind that meshlearn is in the alpha stage, use in production is not yet recommended. You are free to play around with it though!**

Currently meshlearn comes with one pre-trained model for predicting the local gyrification index (lGI, Schaer et al.) for full-resolution, native space [FreeSurfer meshes](https://freesurfer.net/). These meshes are (a part of) the result of running FreeSurfer's `recon-all` pipeline on structural MRI scans of the human brain.

The model is a gradiant-boosting machine as implemented in [lightgbm](https://github.com/microsoft/LightGBM), and it was trained on a diverse training set of about 60 GB of pre-processed mesh data, obtained from the publicly available, multi-site [ABIDE I dataset](https://fcon_1000.projects.nitrc.org/indi/abide/). The model can be found at [tests/test_data/models/lgbm_lgi/](./tests/test_data/models/lgbm_lgi/), and consists of the model file (`ml_model.pkl`, the pickled lightgbm model) and a metadata file (`ml_model.json`) that contains the pre-processing setting used to train the model, that must also be used when predicting for a new mesh.

The `meshlearn_lgi_predict` command line application that is part of meshlearn can be used to predict lGI for your own FreeSurfer meshes using the supplied model or alternative models. After installation of meshlearn, run `meshlearn_lgi_predict --help` for available options.

If you want to train your own model, you will need training data and a powerful machine with 128+ GB of RAM. Please following the development instructions for details.

## Development

### Development state

**This currently is a quick prototype and not intended to be used by others. There is no stable API whatsoever, everything changes at will.**

### Development installation for Ubuntu 20.04 LTS and Ubuntu 22.04

This will most likely work under a number of different operating systems, but Ubuntu 20.04 LTS and Ubuntu 22.04 are the only ones I tested.

We highly recommend to work in a `conda` environment, especially when using `tensorflow-gpu` instead of the CPU-version `tensorflow`:


#### Step 1 of 2: Create conda env and install conda packages into it

If you want to run the neural network scripts that use tensorflow and you have a powerful GPU, I highly recommend that you install `tensorflow-gpu` to use it. Here is how I did it under Ubuntu 20.04 LTS:

```shell
conda create -y --name meshlearn python=3.7
conda activate meshlearn
conda install -y tensorflow-gpu  # Or just 'tensorflow' if you don't have a suitable GPU.
conda install -y pandas matplotlib ipython scikit-learn psutil lightgbm
conda install -y -c conda-forge scikit-learn-intelex  # Not strictly needed, speedups for scikit-learn.
conda install -y -c conda-forge trimesh igl
```

If you do not have a suitable GPU, simply replace `tensorflow-gpu` with `tensorflow` to run on CPU.

Alternatively, one can use the [environment.yml file we now provide](./environment.yml) to setup the conda environment. This is guaranteed to be up-to-date and to work, as this is the way we also install on our [CI workflow](.github/workflows/cov_test_workflow.yml) where the unit tests are run.



#### Step 2 of 2: Install meshlearn into the conda env ####

Checkout the repo using git:

```bash
conda activate meshlearn  # If not done already.
git clone https://github.com/dfsp-spirit/meshlearn
cd meshlearn
```

Then install:

```bash
pip3 install --upgrade pip
pip3 install -e .
```

### Running the development version


#### Obtaining training data

##### Option 1: Generating your own training data

With some computational resources and experience with structural neuroimaging, you can generate your own training data:

* Download a suitable, large collection of T1-weighted (and optionally T2-weighted) structural MRI scans from many healthy subjects. To avoid bias, only use controls in case its a clinical dataset. Make sure to include subjects from as many sites (different scanners) as possible, as well as a wide age range, different genders, etc.
     - An option is to use all controls from the ABIDE dataset.
     - The more sites and subjects, the better. We suggest at least 20 sites and 300 subjects.
     - Consider excluding bad quality scans.
* Pre-process all scans with FreeSurfer v6 (full recon-all pipeline). This takes about 12 - 18 hours per subject when done sequentially on a single core of a 2022 consumer desktop computer.
     - When pre-processing is done, compute pial-lgi for all subjects.

##### Option 2: Downloading our training data

We now make our training data publicly available. See the [native space lgi data for all ABIDE I subjects](https://doi.org/10.5281/zenodo.7132610) on Zenodo (6.5 GB download).


#### Running model training

Have a look at the `meshlearn_lgi_train` application and its command line options. After installation of the `meshlearn` Python package (see above for instructions), just type `meshlearn_lgi_train --help` to get started.

If you need more control, use the meshlearn Python API. We would suggest to have a look at our work to get started: select the model you want to run in `src/clients/` and create your own copy. Then adapt the settings at the top and/or the defaults for the command line arguments or mess with the code.

Note: Be sure to run within the correct `conda` environment!


#### Running the unit tests

See [tests/meshlearn/](./tests/meshlearn/) for the unit tests. To run them, you will need to have `pytest`` installed in your environment. If you do not have that already, first install it:

```shell
conda activate meshlearn
conda install -y pytest
```
Then run the tests:

```shell
cd <repo_dir>
export PYTHON_PATH=$(pwd)
cd tests/
pytest
```

#### Test coverage

The test coverage on the CI system seems quite low, but the reason is that GitHub CI cannot run large parts of the tests due to the very limited amount of memory (see [here](https://codecov.io/github/dfsp-spirit/meshlearn)), which shows 49% coverage at the time of this writing, while I have 71% coverage locally.

To see realistic coverage while working on the code, I would currently recommend to run it locally, in your dev installation. E.g.:

```shell
conda activate meshlearn
pip install pytest-cov  # If you don't have it yet.
cd <repo_dir>
pytest --cov-report term:skip-covered --cov src/meshlearn tests/
```

If you prefer a full HTML report (more or less required if you want to add new tests specifically designed to increase coverage), create it instead of the console report:

```shell
pytest --cov-report html:coverage_html --cov src/meshlearn tests/
firefox ./coverage_html/index.html  # Or whatever your favourite browser is.
```
