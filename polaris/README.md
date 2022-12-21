# Polaris Beginners Guide

This guide will teach the basics of how to utilize Polaris to achieve your scientific goals. 


## [Polaris](https://www.alcf.anl.gov/polaris)

![Polaris](media/polaris.jpg)

The inside of Polaris again shows the _nodes_ stacked up in a closet.

![Polaris-rack](media/polaris1.jpg)

Polaris is an NVIDIA A100-based system.

Polaris Machine Specs
* Speed: 44 petaflops
* Each Node has:
  * 4 NVIDIA (A100) GPUs
  * 1 AMD EPYC (Milan) CPUs
* ~560 Total Nodes


## Logging in:

Login using `ssh` replacing `<username>` with your ALCF username
```bash
ssh <username>@polaris.alcf.anl.gov
```

![login](media/polaris_login.gif)

You will be prompted for your password, which is a six digit code generated uniquely each time using the MobilePASS+ app. 

## Quick filesystem breakdown

When you login, you start in your _home_ directory: `/home/<username>/` (100GB quota)
As an ALCF user you will be assigned access to different allocation _projects_. You can see your projects listed on the [ALCF Accounts Page](accounts.alcf.anl.gov). Each project maps to a user group to control filesystem access, so you can also check your projects using the `groups` command on the terminal. Projects are given storage spaces on our Eagle and/or Grand Lustre filesystems where all members of the project can read/write and share data/software:
* `/eagle/<project-name>`
* `/grand/<project-name>`
Users should use project spaces for large scale storage and software installations. Increases can be requested via `support@alcf.anl.gov`.

## Clone repo:

Next, clone this repository into your home directory using:
```bash
git clone https://github.com/argonne-lcf/ALCFBeginnersGuide.git
```

![clone](media/polaris_git_clone_repo.gif)

