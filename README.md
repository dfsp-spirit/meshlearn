# meshlearn
AI model to predict computationally expensive local, vertex-wise descriptors like the local gyrification index from the local mesh neighborhood.

**This currently is a quick prototype and not intended to be used by others. There is no stable API whatsoever, everything changes at will.**


## About

Predict per-vertex descriptors like the local gyrification index (lGI) or other local descriptors for a mesh.

* The local gyrification index is a brain morphometry descriptor used in computational neuroimaging. It describes the folding of the human cortex at a specific point, based on a mesh reconstruction of the cortical surface from a magnetic resonance image (MRI). See [Schaer et al. 2008](https://doi.org/10.1109/TMI.2007.903576) for details.
* The geodesic circle radius and related descriptors are described in my [cpp_geodesics repo](https://github.com/dfsp-spirit/cpp_geodesics) and in the references listed there. Ignore the global descriptors (like mean geodesic distance) in there.


![Vis1](./web/brain_mesh_full.jpg?raw=true "Brain mesh, white surface.")

**Fig. 1** *A mesh representing the human cortex.*

![Vis2](./web/brain_mesh_vertices.jpg?raw=true "Brain mesh, zoomed view that shows the mesh structure.")

**Fig. 2** *Close up view of the triangular mesh, showing the vertices, edges and faces. Each vertex neighborhood (example for the ML model) describes the mesh structure in a sphere around the respective vertex. Vertex neighborhoods are computed from the mesh during pre-processing.*

This implementation uses Python, with `tensorflow` and `lightgbm` for the machine learning part. Mesh pre-processing is done with `pymesh` and `igl`.

## Why

Computing lGI and some other mesh properties for brain surface meshes is slow and sometimes fails even for good quality meshes, leading to exclusion of the respective MRI scans. The lGI computation also requires Matlab, which is inconvenient and prevents the computation of lGI on high performance computer clusters (due to the excessive licensing costs), which would be a way to deal with the long computation times. This project aims to provide a trained model that will predict the lGI for a vertex based on the mesh neighborhood. The aim is to have a faster and more robust method to compute lGI, based on free software.

## Development

Note: This is structured like a python module, but the code should be treated as a very specific application, I guess. It's just convenient for me to have it in a model to re-use some data loading stuff.


### Development installation for Ubuntu 20.04 LTS

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

If you do not have a suitable GPU, simply replace `tensorflow-gpu` with `tensorflow`.

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

Select the model you want to run in `src/clients/` and adapt the settings at the top and/or the defaults for the command line arguments.

Then run the client script in `ipython` or use one of the run scripts, like: `./run_lgbm.sh`, after adapting the command line arguments in there.

Note: Be sure to run within the correct `conda` environment!


#### Running the unit tests

See [./tests/meshlearn/](./tests/meshlearn/) for the unit tests. To run them, you will need to have `pytest`` installed in your environment. If you do not have that already, first install it:

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
