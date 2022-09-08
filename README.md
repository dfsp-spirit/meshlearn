# meshlearn
AI model to predict computationally expensive vertex-wise descriptors like the local gyrification index from the mesh structure.


## About

Predict vertex descriptors like the local gyrification index (lGI) or other local descriptors for a mesh vertex.

* The local gyrification index is a brain morphometry descriptor used in computational neuroimaging. It describes the folding of the human cortex at a specific point, based on a mesh reconstruction of the cortical surface from a magnetic resonance image (MRI). See [Schaer et al. 2008](https://doi.org/10.1109/TMI.2007.903576) for details.
* The geodesic circle radius and related descriptors are described in my [cpp_geodesics repo](https://github.com/dfsp-spirit/cpp_geodesics) and in the references listed there.


![Vis1](./web/brain_mesh_full.jpg?raw=true "Brain mesh, white surface.")

**Fig. 1** *A mesh representing the human cortex.*

![Vis2](./web/brain_mesh_vertices.jpg?raw=true "Brain mesh, zoomed view that shows the mesh structure.")

**Fig. 2** *Close up view of the triangular mesh, showing the vertices, edges and faces.*

This implementation uses Python/tensorflow.

## Why

Computing lGI and some other mesh properties for brain surface meshes is slow and sometimes fails even for good quality meshes, leading to exclusion of the respective MRI scans. The lGI computation also requires Matlab, which is inconvenient and prevents the computation of lGI on high performance computer clusters (due to the excessive licensing costs), which would be a way to deal with the long computation times. This project aims to provide a trained model that will predict the lGI for a vertex based on the mesh neighborhood. The aim is to have a faster and more robust method to compute lGI, based on free software.

## Development

### Development setup for Ubuntu >= 19.04

Checkout the repo using git:

```bash
git clone https://github.com/dfsp-spirit/meshlearn
cd meshlearn
```

Now get required pypi packages:

```bash
pip3 install --upgrade pip
pip3 install -e .
```

### Running the unit tests

```bash
#pip3 install pytest
cd <repo_dir>
python3 -m pytest tests/
```

### Getting tensorflow-gpu stuff running

If you want to run the neural network scripts that use tensorflow and you have a powerful GPU, I highly recommend that you install `tensorflow-gpu` to use it. Here is how I did it under Ubuntu 20.04 LTS:

```shell
conda create --name meshlearn-gpu python=3.7
conda activate meshlearn-gpu
conda install tensorflow-gpu
conda install pandas
conda install matplotlib
conda install ipython
conda install scitkit-learn
```

Keep in mind though that your GPU memory may be smaller than your system RAM, and you will most likely have to train in batches.
