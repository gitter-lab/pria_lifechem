# Virtual screening on PriA-SSB and RMI-FANCM with the LifeChem library

[![Build Status](https://travis-ci.org/gitter-lab/pria_lifechem.svg?branch=master)](https://travis-ci.org/gitter-lab/pria_lifechem)

## Citation

If you use this software or the new high-throughput screening data, please cite:

Shengchao Liu<sup>+</sup>, Moayad Alnammi<sup>+</sup>, Spencer S. Ericksen, Andrew Voter, James Keck, F. Michael Hoffmann, Scott A. Wildman, Anthony Gitter.
Practical model selection for prospective virtual screening.
2018.

This manuscript will soon be posted on *bioRxiv*.
<sup>+</sup> denotes co-first authors.

## Installation

We recommend creating a [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) to manage the dependencies.
First [install Anaconda](https://www.anaconda.com/download/) if it is not already installed.
Then, clone this `pria_lifechem` repository:
```
git clone https://github.com/gitter-lab/pria_lifechem.git
cd pria_lifechem
```

Create and activate a conda environment named `pria` using the `conda_env.yml` file:
```
conda env create -f conda_env.yml
source activate pria
```

Finally, install `pria_lifechem` with `pip`.
```
pip install -e .
```

To use the package again later, use `source activate pria` to re-activate the conda environment.
The package is only currently supported for Linux.
The conda environment provided does not include a Theano GPU backend.
To use Theano with a GPU, see the [Theano guide](http://deeplearning.net/software/theano_versions/0.8.X/tutorial/using_gpu.html).

The IRV models were trained using a customized [fork of DeepChem](https://github.com/Malnammi/deepchem).
See the separate installation instructions in that repository.

Note: Random Forest results in the paper were obtained using Python 3.4 and sklearn=0.18.1.
The random forest code is still compatible with `conda_env.yml`, but the results may differ due to different versions.

## dataset

The dataset subdirectory contains a description of the expected file format and an example dataset that has been split into five folds.

The complete high-throughput screening data will be uploaded to PubChem.
Pre-processed, merged versions of the data will be available on Zenodo.
The Zenodo files contain:
- The LifeChem compounds used for cross validation with PriA-SSB and RMI-FANCM labels split into five folds.
- These same compounds merged with 128 tasks from PubChem split into five folds.
- The separate LifeChem componds used for prospective testing with PriA-SSB.

## pria_lifechem

The pria_lifechem subdirectory contains:

- scripts to prepare and load datasets
- a script to evaluate trained models
- a models subdirectory with code and instructions for training models
- an analysis subdirectory to reproduce figures from the manuscript

## json

The json subdirectory contains json config files with the model hyperparameters.

## output

The output subdirectory contains scripts for post-processing the output files.
