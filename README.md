# lgilearn
AI model to predict the local gyrification index for a vertex based on the mesh instead of computing it.


## About

Predict the local gyrification index (lGI) for a mesh vertex. The local gyrification index is a brain morphometry descriptor used in computational neuroimaging. It describes the folding of the human cortex at a specific point, based on a mesh reconstruction of the cortical surface from a magnetic resonance image (MRI).

![Vis1](https://github.com/dfsp-spirit/lgilearn/blob/master/web/brain_mesh_full.png?raw=true "Brain mesh, white surface.")
![Vis2](https://github.com/dfsp-spirit/lgilearn/blob/master/web/brain_mesh_vertices.png?raw=true "Brain mesh, zoomed view that shows the mesh structure.")


## Why

Computing lGI for brain surface meshes is slow and sometimes fails even for good quality meshes, leading to exclusion of the respective MRI scans. It also equires Matlab, which is inconvenient and typically prevents the computation of lGI on computer clusters (due to the excessive costs), which would be a way to deal with the long computation times. This project aims to provide a trained model that will predict the lGI for a vertex based on the mesh neighborhood. The aim is to have a faster and more robust method to compute lGI, using only free software.

## Development

### Development setup for Ubuntu >= 19.04

Checkout the repo using git:

```bash
git clone https://github.com/dfsp-spirit/lgilearn
cd lgilearn
```

Now get required pypi packages:

```bash
pip3 install --upgrade pip
pip3 install .
```
