# Using the Polaris Job Scheduler: PBS

Polaris is the first machine at ALCF to use the PBS scheduler. PBS is a third party product that comes with [extensive documentation](https://help.altair.com/2022.1.0/PBS%20Professional/PBSUserGuide2022.1.pdf). This is an introduction, not an extensive tutorial so we will only cover some basics.

## What is a scheduler

A _scheduler_ is used to fairly run applications on a large number of computers that are shared by many users. The user installs their software and data in a space accessible by the worker nodes, then creates a script (BASH or otherwise) that goes through the motions of running the application. Based on the user's needs, they submit a _job_ to the _scheduler_ that defines the number of compute nodes needed and the length of time the job should run for, also called _wall-time_. Given this information, the schduler pieces together all jobs in an efficient and fair way to run them all.

## Running interactively

When you login to the supercomputer, you are given a shell running on one of the few _login nodes_. These are shared with every other user logged into the system at that time, so they are not meant for running compute intensive things. If you would like to build software or make test runs on an actual _worker_ node, please start an interactive session in the following way:

```bash
qsub -I -l select=1 -l walltime=00:30:00 -q debug -l filesystems=home -A <project-name>
```

Here are the command breakdown:
* `qsub` is the command to submit jobs to the scheduler
* `-I` means submit an _interactive_ job
* `-l select=1` means we want one compute node for this job
* `-l walltime=00:30:00` means we want our one node for 30 minutes (format = "HOURS:MINUTES:SECONDS")
* `-q debug` tells the scheduler which _queue_ we would like to use
* `-l filesystems=home` tells the scheduler that we require our home directory for this job. You can also specify `filesystems=home:eagle` if you also need access to `/eagle/<project-name>/` or `filesytems=home:grand` for `/grand/<project-name>/`.
* `-A <project-name>` specifies the project to which this job will be charged

After your job begins, you will be running a shell on a worker node. The environment can be setup using `module` and some things are already loaded, including some NVidia tools like `nvidia-smi`.

![polaris_interactive](media/polaris_qsub_interactive.gif)

## Submit your first job

The more standard method for running a job is to submit it to the scheduler via `qsub` with a script that will execute your job. Let's walk through an example.

First we need to create a job script (example: [submit_scripts/00_hello_world.sh](submit_scripts/00_hello_world.sh)):
```bash
#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A <project-name>
#PBS -o logs/
#PBS -e logs/

module load gcc/11.2.0

GPUS_PER_NODE=4

mpiexec -n $GPUS_PER_NODE -ppn $GPUS_PER_NODE echo Hello World

```

You'll notice we can use the `#PBS` line prefix at the top of our script to set `qsub` command line options. We can still use the command line to override the options in the script. 
NOTE: here we used `-o logs/` and `-e logs/` which just redirects the STDOUT(`-o`) and the STDERR(`-e`) log files from the job into the `logs/` directory to keep things tidy.

Note: Job scripts must be executable so we need to run `chmod a+x job_script.sh`.

Now submit our job (don't forget to change `<project-name>`):
```bash
qsub job_script.sh
```

## Monitor your job

We can check our job's status using this command:
```bash
qstat -u <username>
```
without specifying the `username` we will get a full printout of every job queued and running. This can be overwhelming so using the `username` reduces the output to jobs for just that `username`.

![polaris_hello_world](media/polaris_qsub_hello_world.gif)
