# Python Environments

### Users are assumed to know:
* how to use Python
* basic Conda usage
### Learning Goals:
* How to add prebuilt Python environments into your environment
* Loading a Conda module
* Create a custom build environment based on pre-existing Conda module

## Overview

ALCF provides pre-built Python environments using `miniconda`. Within these environments, ALCF compiles GPU-supported python libraries, such as:

- [TensorFlow](https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/tensorflow/)
  - [Horovod](https://horovod.readthedocs.io/en/stable/tensorflow.html)
- [PyTorch](https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/pytorch/)
  - [DDP](https://pytorch.org/tutorials/beginner/dist_overview.html)
  - [Horovod](https://horovod.readthedocs.io/en/stable/pytorch.html)
  - [DeepSpeed](https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/deepspeed/)
- [JAX](https://jax.readthedocs.io/en/latest/)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)

## Loading Python Environment

Remember you can list all the available Conda environments using `module list conda`. As of this writing there is only one, `conda/2024-04-29`. ALCF typically installs an environment every six months including the latest versions of Tensorflow and PyTorch (built from source). The date signifies when it was built.

To load and activate the default environment:

```Shell
# first tell modules where to find conda
module use /soft/modulefiles
# load conda into your environment
module load conda
# activate the `base` conda environment
conda activate base
```

If you need to install additional packages, there are two approaches covered in the following sections:

1. [Virtual environments via `venv`](#virtual-environment-via-venv): builds an extendable enviroment on top of the immutable base environment.
2. [Clone the base `conda` environment](#clone-conda-environment): complete mutable copy of the base environment into a user's space.

In general, these are things that should be done in a user's project directory to get the best performance and available capacity.

## Virtual Environment via `venv`

The easiest method for making a custom environment that builds on-top of the ALCF environment is to use `venv`. 

```Shell
python -m venv /path/to/venvs/base --system-site-packages
```

By passing the `--system-site-packages` flag, we are able to create our own
isolated environment into which we can install new packages, while taking
advantage of the pre-built libraries from `conda`.

To activate this new environment,

```Shell
source /path/to/venvs/base/bin/activate
```

Once activated, installing packages with pip is as usual:

```Shell
python -m pip install <new-package>
```

To install a _different version_ of a package that is **already installed** in the
base environment add the `--ignore-installed` to your command:

```Shell
python -m pip install --ignore-installed <new-package>
```

## Clone Conda Environment

Cloning a Conda Environment creates a full copy of the ALCF conda environment in a specified directory. This means the user has full control of the environment. 

This process takes 11 GB of disk space and 15 minutes to complete.

Create a `clone` of the base environment by:

```Shell
# load conda
module load conda ; conda activate base
# create the clone
conda create --clone base --prefix /path/to/envs/myclone
# load the cloned environment
conda activate /path/to/envs/myclone
```

Future loading can be done with the following. It is necessary to ensure the same version of conda you are loading is the same as that with which you generated the clone.

```Shell
module load conda; conda activate /path/to/envs/myclone
```


## Additional Resources

- [ALCF Docs: Python on Polaris](https://docs.alcf.anl.gov/polaris/data-science-workflows/python/)


# [NEXT ->](04_jupyterNotebooks.md)
