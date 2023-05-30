# Python Environments

### Users are assumed to know:
* how to use Python
* basic Conda usage
### Learning Goals:
* Loading a Conda module on Polaris
* Create a custom build environment based on a Conda module


<details closed><summary><b>ℹ️ Note: Job Setup</b></summary>
<p>
  
> **Note**
> <br> The instructions below should be **ran directly from a compute node**.
>
> Explicitly, to request an interactive job (from `polaris-login`):
> ```Shell
> qsub -A <project> -q debug-scaling -l select=2 -l walltime=01:00:00 -I
> ```
> Refer to
> [job scheduling and execution](https://docs.alcf.anl.gov/running-jobs/job-and-queue-scheduling/)
> for additional information.
>

> **Warning**
> <br> In addition, all example paths specified below should be replaced by a
> suitably chosen destination.
>
> For example,
> ```Shell
> /path/to/miniconda3/envs/
> ```
> would be replaced by
> ```Shell
> /lus/grand/projects/datascience/foremans/miniconda3/envs/
> ```

</p>
</details>

We provide pre-built `conda` environments, ready-to-go with your all[^all] your favorite GPU-supported python libraries:

- [TensorFlow](https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/tensorflow/) **\+**
  - [Horovod](https://horovod.readthedocs.io/en/stable/tensorflow.html)
- [PyTorch](https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/pytorch/) **\+**
  - [DDP](https://pytorch.org/tutorials/beginner/dist_overview.html)
  - [Horovod](https://horovod.readthedocs.io/en/stable/pytorch.html)
  - [DeepSpeed](https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/deepspeed/)
- [JAX](https://jax.readthedocs.io/en/latest/)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)

To load and activate the default[^versions] `conda` environment[^conda1], simply:

```Shell
module load conda ; conda activate base
```

If you need to install additional packages, there are two approaches:

1. [Virtual environments via `venv`](#virtual-environment-via-venv)
2. [Clone the base `conda` environment](#clone-conda-environment)

## Virtual Environment via `venv`

The most straightforward approach uses `venv` to create a new virtual
environment _on top of_ our existing (`conda`) environment.

By passing the `--system-site-packages` flag, we are able to create our own
isolated environment into which we can install new packages, while taking
advantage of the pre-built libraries from `conda`.

```Shell
python3 -m venv /path/to/venvs/base --system-site-packages
```

To activate this new environment,

```Shell
source /path/to/venvs/base/bin/activate
```

To install a different version of a package that is already installed in the
base environment,

```Shell
python3 -m pip install --ignore-installed <new-package>
```

## Clone Conda Environment

If you need additional packages that require a `conda install`, you can create
a `clone` of the base environment by:

```Shell
module load conda ; conda activate base
conda create --clone base --prefix /path/to/envs/base-clone
conda activate /path/to/envs/base-clone
```

> **Warning**
> <br> Creating a clone of the base enviroment
> can be quite slow (and large).
>
> It is recommended to create this clone somewhere
> outside of your `$HOME` directory, if possible.
>
> For example,
> `mkdir -p "/grand/<project>/${USER}/miniconda3/envs"`


## Additional Resources

- [ALCF Docs: Python on Polaris](https://docs.alcf.anl.gov/polaris/data-science-workflows/python/)


[^versions]:
    As of writing, the latest `conda` module on Polaris is built on Miniconda3
    version 4.14.0 and contains Python 3.8.13.
[^all]:
    Additional packages can be installed either in a virtual environment, or
    inside a clone of the base environment.
[^conda1]:
    You can get a list of all available `conda` modules with `module avail conda`.
