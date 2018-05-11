# Virtual screening on PriA-SSB and RMI-FANCM with the LifeChem library

[![Build Status](https://travis-ci.org/gitter-lab/pria_lifechem.svg?branch=master)](https://travis-ci.org/gitter-lab/pria_lifechem)

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

## dataset

The high-throughput screening data will be uploaded to PubChem.
Pre-processed, merged versions of the data will be available on Zenodo.
We have a pre-fixed split dataset for LC_Pria, PCBA, and LC_PCBA combined.

keck_updated_complete.csv: contains complete data for Keck_Pria

## pria_lifechem

- models:

  - `deep_classification.py` SingleClassification (STNN-C)
  
  - `deep_regression.py` SingleRegression (STNN-R)
  
  - `deep_classification.py` MultiClassification (MTNN-C)
  
  - `vanilla_lstm.py` VanillaLSTM (LSTM)
  
  - `sklearn_randomforest.py` RandomForest (RF)
  
  - `deepchem_irv_single.py` InfluenceRelevanceVoter (IRV)

  - `stage_hyperparameter_search.py`
  
  - `stage_cross_validation.py`
  
  - `stage_prospective_screening.py`

- `data_preparation.py`

- `function.py`

- `evaluation.py`

- `integrity_checker.py`

- analysis

## json

This subdirectory contains json config files with the model hyperparameters.

## output

This subdirectory contains scripts for post-processing the output files.
