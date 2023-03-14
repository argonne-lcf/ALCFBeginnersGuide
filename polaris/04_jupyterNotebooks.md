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
# set shell proxy variables to access external URL
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

From within the JupyterHub file browser, users are limited to viewing files
within their home directory.

To access project directories located outside of your `$HOME`, a symbolic link
to the directory must be created.

Explicitly, if a user wants to access project `ABC`, we can create a symbolic
link by

<details open>
<summary>
<b>
From Terminal
</b>
</summary>
<p>
 
```Shell
# from terminal
cd ~
ln -s /grand/projects/ABC ABC_project
```
</p>
</details>

<details closed>
<summary>
<b>
From Notebook
</b>
</summary>
<p>

```Shell
# from notebook
!ln -s /grand/projects/ABC ABC_project
```
</p>
</details>

## Running Notebook on a Compute Node

The ThetaGPU and Polaris instances of JupyterHub allow users to start Jupyter
Notebooks directly on compute nodes through the given job scheduler.

The job will be executed according to ALCF's queue and scheduling policy[^timeout]

## Polaris

The Polaris JupyterHub instance does not have a "Local Host Process" option.

All jupyter notebooks are run on a compute node through the job scheduler.

When the user authenticates the user will be presented with a "Start My Server"
button that once clicked, will present the user with the available job options
needed to start the notebook.

- Options:
  - Select a job profile: This field lists the current available Profiles
    "Polaris Compute Node", etc.
  - Queue Name: This field provides a list of available queues on the system
  - Project List: This field displays the list of active projects associated
    with the user on the given system
  - Number Chunks: This field allows the user to select the number of compute nodes to be allocated for the job
  - Runtime (minutes:seconds): This field allows the user to set the runtime of the job in minutes and seconds.
  - File Systems: This field allows the user to select which file systems are required.


[^docs]: Additional information can be found in our [JupyterHub documentation](https://docs.alcf.anl.gov/services/jupyter-hub/)
[^timeout]: Note: if the queued job does not start within 2 minutes, JupyterHub
  will timeout and the job will be removed from the queue.
