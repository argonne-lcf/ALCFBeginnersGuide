# PyTorch on Polaris

PyTorch is a popular, open-source deep learning framework developed and released by Facebook. The [PyTorch home page](https://pytorch.org/) has more information about PyTorch, which you can refer to. For troubleshooting on Polaris, please contact support@alcf.anl.gov.

## Installation on Polaris

PyTorch is installed on Polaris already, available in the `conda` module. To use it from a compute node, please do:

```bash
module use /soft/modulefiles
module load conda
conda activate
```

Then, you can load PyTorch in `python` as usual (below showing results from the `conda/2024-04-29` module):

```python
>>> import torch
>>> torch.__version__
'2.3.0'
>>>
```

This installation of PyTorch was built from source, and the CUDA libraries it uses are found via the `CUDA_HOME` environment variable (below showing results from the `conda/2024-04-29` module):

```bash
$ echo $CUDA_HOME
/soft/compilers/cudatoolkit/cuda-12.4.1/
```

If you need to build applications that use this version of PyTorch and CUDA, we recommend using these CUDA libraries to ensure compatibility. We periodically update the PyTorch release, though updates will come in the form of new versions of the `conda` module.

PyTorch is also available through NVIDIA containers that have been translated to Apptainer containers. For more information about containers, please see the [containers](../../containers/containers.md) documentation page.

## PyTorch Best Practices on Polaris

### Single Node Performance

When running PyTorch applications, we have found the following practices to be generally, if not universally, useful and encourage you to try some of these techniques to boost the performance of your own applications.

1. Use Reduced Precision. Reduced Precision is available on A100 via tensor cores and is supported with PyTorch operations. In general, the way to do this is via the PyTorch Automatic Mixed Precision package (AMP), as described in the [mixed precision documentation](https://pytorch.org/docs/stable/amp.html). In PyTorch, users generally need to manage casting and loss scaling manually, though context managers and function decorators can provide easy tools to do this.

2. PyTorch has a `JIT` module as well as backends to support op fusion, similar to TensorFlow's `tf.function` tools. However, PyTorch JIT capabilities are newer and may not yield performance improvements. Please see [TorchScript](https://pytorch.org/docs/stable/jit.html) for more information.

## Multi-GPU / Multi-Node Scale up

PyTorch is compatible with scaling up to multiple GPUs per node, and across multiple nodes. Good scaling performance has been seen up to the entire Polaris system, > 2048 GPUs. Good performance with PyTorch has been seen with both DDP and Horovod. For details, please see the [Horovod documentation](https://horovod.readthedocs.io/en/stable/pytorch.html) or the [Distributed Data Parallel documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html). Some Polaris-specific details that may be helpful to you:

<!-- --8<-- [start:scalingsetup] -->
1. CPU affinity can improve performance, particularly for data loading processes. In particular, we encourage users to try their scaling measurements by manually setting the CPU affinity via mpiexec, such as with `--cpu-bind verbose,list:0,8,16,24` or `--cpu-bind depth -d 16`.

2. NCCL settings: 
--8<-- "./docs/polaris/applications-and-libraries/libraries/nccl.md:ncclenv"

3. CUDA device setting: it works best when you limit the visible devices to only one GPU. Note that if you import `mpi4py` or `horovod`, and then do something like `os.environ["CUDA_VISIBLE_DEVICES"] = hvd.local_rank()`, it may not actually work! You must set the `CUDA_VISIBLE_DEVICES` environment variable prior to doing `MPI.COMM_WORLD.init()`, which is done in `horovod.init()` as well as implicitly in `from mpi4py import MPI`. On Polaris specifically, you can use the environment variable `PMI_LOCAL_RANK` (as well as `PMI_LOCAL_SIZE`) to learn information about the node-local MPI ranks.  
<!-- --8<-- [end:scalingsetup] -->

### DeepSpeed

DeepSpeed is also available and usable on Polaris. For more information, please see the [DeepSpeed](./deepspeed.md) documentation directly.

## PyTorch `DataLoader` and multi-node Horovod

For best performance, it is crucial to enable multiple workers in the data loader to avoid compute and I/O overlap and concurrent loading of the dataset. This can be set by tuning the "num_workers" parameter in ```DataLoader``` (see https://pytorch.org/docs/stable/data.html). According to our experience, generally, one can set 4 or 8 for best performance. Due to the total number of CPU cores available on a node, the maximum number of workers one can choose is 16. It is always best to tune this value and find the optimal setup for your own application.

Aside from this, one also has to make sure that the worker threads spread over different CPU cores. To do this, one has to specify the CPU binding to be `depth` and choose a depth value larger than ```num_workers``` through the following flag in the ```mpiexec``` command:

```
mpiexec -np $NUM_GPUS -ppn 4 --cpu-bind depth -d 16 python3 ...
```

Before 2024, enabling multiple workers would cause a fatal hang, but this has been addressed after an OS upgrade on Polaris.
# [NEXT ->](05_jupyterNotebooks.md)
