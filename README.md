# lgi_learn
AI model to predict the local gyrification index for a vertex based on the mesh instead of computing it.


## About

Computing lGI for brain surface meshes is slow and sometimes fails even for good quality meshes, leading to exclusion of the respective MRI scans. This project aims to provide a trained model that will predict the lGI for a vertex based on the mesh neighborhood.

## Development

### Development setup for Ubuntu >= 19.04

Checkout the repo using git:

```bash
cd ~/develop/    # Or whereever you want to put it.
git clone https://github.com/dfsp-spirit/lgi_learn
cd lgi_learn
```

Now get required pypi packages:

```bash
pip3 install --upgrade pip
pip3 install tensorflow
```
