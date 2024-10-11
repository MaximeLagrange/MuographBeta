[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# MuographBeta: muon tomography library

![logo](./images/muograph_logo.png)


This repo provides a library for muon scattering tomography and muon transmission tomography data analysis. 

## Overview

As a disclaimer, this library is more of an aggregate of muon tomography algorithms used throughtout PhD research rather than a polished product for the general public. As such, this repo targets mostly muon tomography reaserchers and enthousiasts.

Users can find ready to use scattering density algorihms as well as samples of simulated data.

While curently being at a preliminary stage, this library is designed to be extended by users, whom are invited to implement their favorite reconstruction, material inference or image processing algorithms.

![image](./images/mst_image_example.png)

## Requierments

The Python libraries required can be installed using [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html), a powerful command line tool for package and environment managment.

It can be installed following these [instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), based on your operating system.

Simply run the following command:

```bash
cd MuographBeta/
conda env create --file=environment.yml
```

You can then activate/deactivate the environment with:

```bash
conda activate muograph
```

```bash
conda deactivate
```

## Installation

Now that the required libraries have been installed, the repository can be cloned:

```bash
git clone https://github.com/MaximeLagrange/MuographBeta.git
```

This repo Git Large File Storage ([LFS](https://git-lfs.com/)) to handle the data files that are too large for GitHub. To ensure that all LFS files are downloaded, run the following command inside the repository folder:

```bash
git lfs pull
```

## Examples

A few examples to introduce users to the package can be found in the `./example/` folder.

![example](./images/examples_preview.png)
