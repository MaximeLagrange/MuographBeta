[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# MuographBeta: reconstruction software

This repo provides a library for muon scattering tomography and muon transmission tomography data analysis. 

## Overview

As a disclaimer, this library is more of an aggregate of muon tomography algorithms used throught PhD research rather than a polished product for the general public. As such, this repo targets mostly muon tomography reaserchers and enthousiasts.

While cuurently being at a preliminary stage, his library is designed to extended by users, whom are invite to implement their favorite reconstruction, material inference or image processing algorithms.


## Installation

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


## Examples

A few examples are included to introduce users to the package:

 - 01_MST_reconstruction_example.ipynb
 - 02_generate_voxel_data.ipynb
 - 03_pca_on_mst.ipynb

Example 00 requires the `1M_gen_2hods_30cm_panel_gap_barrel_voi.csv` data file.
Example 01 requires the `no_material_10cm3_1M_gen_new_source.csv` and `iron_10cm3_1M_gen_new_source.csv` data files.
Example 02 requires the `Aluminum.pickle`, `Glass.pickle`, `Iron.pickle`, `Lead.pickle` and `Uranium.pickle` files.

All these files should be stored in `MuographBeta/data/` folder.