# Meshlearn Development Information

See [the meshlearn README](./README.md) for general information on `meshlearn`.

## Development

### Development state

This is not released yet and still a work-in-progress (WIP). There is not stable API yet, and we do not accept issues. Everything may change at will, without prior notice, from version to version.

### Development installation (using conda)

Note: This will most likely work under a number of different operating systems including Linux, Windows and Mac OS, but Ubuntu 20.04 LTS and Ubuntu 22.04 are the only ones we tested.

We highly recommend to work in a [conda](https://www.anaconda.com/products/distribution) environment, especially when using `tensorflow-gpu` instead of the CPU-version `tensorflow`. Currently installing via conda is the only suggested installation path.



#### Step 1 of 2: Checkout the meshlearn repo ####

Checkout the repo using git. We assume you want to put it directly into your user home, into `~/meshlearn`.

```bash
git clone https://github.com/dfsp-spirit/meshlearn ~/meshlearn
```

#### Step 2 of 2: Create conda env and install packages into it

Install all dependencies of meshlearn via the [environment.yml file](./environment.yml). Th file is also used in our [CI workflow](.github/workflows/cov_test_workflow.yml) where the unit tests are run. It is the safest option to ensure you have the right package versions.


```shell
cd ~/meshlearn
conda env create --name meshlearn --file environment.yml
conda activate meshlearn
```


Then install meshlearn from the local clone of the repo:

```bash
cd ~/meshlearn
pip install -e .
```

##### Optional: Extra packages for improved model training performance

If you want to train models yourself and you have a suitable GPU and CPU, you can install `tensorflow-gpu`, and optionally the `scikit-learn-intelex` package to accelerate `scikit-learn`:

```shell
conda activate meshlearn
conda install -y tensorflow-gpu                       # Optional, GPU support for tensorflow.
conda install -y -c conda-forge scikit-learn-intelex  # Optional, speedups for scikit-learn.
```

You will need to read the documentation of these packages and slightly adapt your client code to make sure that they are actually used.



### Running the development version

Note: If you want to predict only, type `meshlearn_lgi_predict --help` to get started. For model training, read on.

#### Obtaining training data for model training

##### Option 1: Generating your own training data

With some computational resources and experience with structural neuroimaging, you can generate your own training data:

* Download a suitable, large collection of T1-weighted (and optionally T2-weighted) structural MRI scans from many healthy subjects. To avoid bias, only use controls in case its a clinical dataset. Make sure to include subjects from as many sites (different scanners) as possible, as well as a wide age range, different genders, etc.
     - An option is to use all controls from the ABIDE dataset, or subjects from the IXI dataset.
     - The more sites and subjects, the better. We suggest at least 20 sites and 300 subjects.
     - Consider excluding bad quality scans.
* Pre-process all scans with FreeSurfer v6 (full `recon-all` pipeline). This takes about 12 - 18 hours per subject when done sequentially on a single core of a 2022 consumer desktop computer.
     - When pre-processing is done, compute pial-lgi for all subjects using `recon-all` (requires Matlab).

##### Option 2: Downloading our training data

We now make our training data publicly available. See the [native space lgi data for all ABIDE I subjects](https://doi.org/10.5281/zenodo.7132610) on Zenodo (6.5 GB download).

The download includes only the files required for meshlearn training, for the ABIDE I dataset. These are the following files per subject:

* `<subject>/surf/lh.pial`
* `<subject>/surf/rh.pial`
* `<subject>/surf/lh.pial_lgi`
* `<subject>/surf/rh.pial_lgi`



#### Running model training

Have a look at the `meshlearn_lgi_train` application and its command line options. After installation of the `meshlearn` Python package (see above for instructions), just type `meshlearn_lgi_train --help` to get started.

If you need more control, use the meshlearn Python API. We would suggest to have a look at our work to get started: select the model you want to run in `src/clients/` and create your own copy. Then adapt the settings at the top and/or the defaults for the command line arguments or mess with the code.

Note: Be sure to run within the correct `conda` environment!


### Unit tests

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

The test coverage on the CI system seems quite low, but the reason is that GitHub CI cannot run large parts of the tests due to the very limited amount of memory (see [here](https://codecov.io/github/dfsp-spirit/meshlearn)), which shows 49% coverage at the time of this writing, while we have 71% coverage locally.

To see realistic coverage while working on the code, we currently recommend to run it locally, in your dev installation. E.g.:

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

### Contributing

If you find an issue, please [report it here](https://github.com/dfsp-spirit/meshlearn/issues) on Github.

Please keep in mind though that this is still WIP, so we do not consider the API stable yet, and we will most likely ignore feature requests at the current time.

Please get in touch by creating an issue first before submitting pull requests.