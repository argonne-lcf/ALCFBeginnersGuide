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

As covered in [03_pythonEnvs](./03_pythonEnvs.md), we recommend using 
virtual environments built on top of our `conda` environments.

We provide below an example of how to set up Jupyter to run with your custom python `venv`.

From a terminal:

```Shell
module load conda ; conda activate base
# source /poth/to/your/venv/bin/activate
# conda activate your_conda_env
# set shell proxy variables to access external URL
python3 -m ipykernel install \
    --user \
    --name=<kernel-name> \
    --display-name=<kernel-display-name> \
    --env PATH "${PATH}" \
    --env LD_LIBRARY_PATH "${LD_LIBRARY_PATH}" \
    --env MPICH_GPU_SUPPORT_ENABLED "${MPICH_GPU_SUPPORT_ENABLED}" \
    --env CONDA_PREFIX "${CONDA_PREFIX}"
# Installed kernelspec <kernel-name> in /path/to/venv/share/jupyter/kernels/<kernel-name>

```

<!--Currently, Polaris compute nodes access the internet through a proxy.

To configure the kernel to use the proxy, add variables `http_proxy`, and
`https_proxy` to the `env` section.

This will allow users to install packages from the notebook using `!conda`
magic commands.-->

We provide a sample configuration below:

```json
{
 "argv": [
  "/lus/grand/projects/datascience/foremans/locations/polaris/projects/saforem2/Megatron-DeepSpeed/venvs/polaris/2023-01-10/bin/python3",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "[Polaris:2023-01-10] MegatronDeepSpeed",
 "language": "python",
 "metadata": {
  "debugger": true
 },
 "env": {
  "PATH": "/lus/grand/projects/datascience/foremans/locations/polaris/projects/saforem2/Megatron-DeepSpeed/venvs/polaris/2023-01-10/bin:/soft/datascience/conda/2023-01-10/mconda3/bin:/soft/datascience/conda/2023-01-10/mconda3/condabin:/soft/compilers/cudatoolkit/cuda-11.8.0/bin:/soft/libraries/nccl/nccl_2.16.2-1+cuda11.8_x86_64/include:/opt/cray/pe/hdf5-parallel/1.12.1.3/bin:/opt/cray/pe/hdf5/1.12.1.3/bin:/opt/cray/pe/pals/1.1.7/bin:/opt/cray/pe/craype/2.7.15/bin:/opt/cray/pe/gcc/11.2.0/bin:/home/foremans/.local/state/fnm_multishells/32267_1680009995525/bin:/home/foremans/.local/state/fnm_multishells/32263_1680009995495/bin:/home/foremans/.fnm:/home/foremans/.linuxbrew/Homebrew/bin:/home/foremans/.linuxbrew/opt/glibc/sbin:/home/foremans/.linuxbrew/opt/glibc/bin:/opt/cray/pe/perftools/22.05.0/bin:/opt/cray/pe/papi/6.0.0.14/bin:/opt/cray/libfabric/1.11.0.4.125/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/home/foremans/bin:/usr/local/bin:/usr/bin:/bin:/opt/c3/bin:/usr/lib/mit/bin:/usr/lib/mit/sbin:/opt/pbs/bin:/sbin:/home/foremans/.linuxbrew/bin:/home/foremans/.linuxbrew/sbin:/home/foremans/.cargo/bin:/home/foremans/.local/bin:/home/foremans/.fzf/bin:/opt/cray/pe/bin",
  "LD_LIBRARY_PATH": "/soft/compilers/cudatoolkit/cuda-11.8.0/extras/CUPTI/lib64:/soft/compilers/cudatoolkit/cuda-11.8.0/lib64:/soft/libraries/trt/TensorRT-8.5.2.2.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/lib:/soft/libraries/nccl/nccl_2.16.2-1+cuda11.8_x86_64/lib:/soft/libraries/cudnn/cudnn-11-linux-x64-v8.6.0.163/lib:/opt/cray/pe/gcc/11.2.0/snos/lib64:/opt/cray/pe/papi/6.0.0.14/lib64:/opt/cray/libfabric/1.11.0.4.125/lib64",
  "MPICH_GPU_SUPPORT_ENABLED": "1",
  "CONDA_PREFIX": "/soft/datascience/conda/2023-01-10/mconda3"
 }
}
```

after completing these steps, you should see `<kernel-name>` kernel when you click
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
