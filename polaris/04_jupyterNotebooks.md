# Jupyter at ALCF

JupyterHub is an open-source application to allow multiple users to launch
Jupyter Notebooks from a central location.

At ALCF, users can use the JupyterHub instances at
[https://jupyter.alcf.anl.gov](https://jupyter.alcf.anl.gov) to run notebooks
directly on the compute resource of their choice[^docs].

We support and maintain a web interface to the [Jupyter Instances at
ALCF](https://jupyter.alcf.anl.gov/) that can be used to run Jupyter notebooks
on all of our major systems.

## Customize Environment

ALCF provides a simple Python environment to start.
Users can customize their environment to meet their needs by creating a virtual
python environment and defining new kernels.

We provide below an example of how to set up a simple environment `projectA`
with module `mpi` from within a notebook.

From a terminal:

```Shell
# Source required conda environment variables from appropriate shell
. /soft/systems/jupyterhub/miniconda3/etc/profile.d/conda.sh
#set shell proxy variables to access external URL
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=$http_proxy

# create an environment name projectA
conda create -y -n projectA

# Activate conda environment
conda activate projectA

# Install required packages
conda install -y jupyter nb_conda ipykernel mpi

# Add environment to available kernel list
python -m ipykernel install --user --name projectA

# deactivate conda environment
conda deactivate
```

Once the base environment is setup, the user must add an `env` section to the
`kernel.json` file, located in `${USER}/.local/share/jupyter/kernels/projecta`,
defining the `$CONDA_PREFIX` and `$PATH` variables.

Currently, Polaris compute nodes access the internet through a proxy.

To configure the kernel to use the proxy, add variables `http_proxy`, and
`https_proxy` to the `env` section.

This will allow users to install packages from the notebook using `!conda`
magic commands.

We provide a sample configuration below:

```json
{
 "argv": [
  "/home/<user>/.conda/envs/projectA/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "projectA",
 "language": "python",
 "env": {
    "CONDA_PREFIX":"/home/<user>/.conda/envs/projecta",
    "PATH":"/home/<user>/.conda/envs/projecta/bin:${PATH}",
    "http_proxy":"http://proxy.alcf.anl.gov:3128",
    "https_proxy":"http://proxy.alcf.anl.gov:3128"
 },
 "metadata": {
  "debugger": true
 }
}
```

after completing these steps, you should see `projectA` kernel when you click
new on the Jupyter Hub home page or when you use Kernel menu in a Jupyter
notebook.

## Accessing Project Folders



[^docs]: Additional information can be found in our [JupyterHub documentation](https://docs.alcf.anl.gov/services/jupyter-hub/)
