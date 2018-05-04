# virtual-screening

[![Build Status](https://travis-ci.com/chao1224/virtual-screening.svg?token=65bvwEHMjNzkwWsY5dLk&branch=master)](https://travis-ci.com/chao1224/virtual-screening)

## Installation

| Package | cmd|
| --- | --- |
| Anaconda | `wget https://repo.continuum.io/archive/Anaconda2-4.3.0-Linux-x86_64.sh -O anaconda.sh` <br> `chmod 777 *` <br> `bash anaconda.sh -b -p $HOME/anaconda` <br> `export PATH="$HOME/anaconda/bin:$PATH"` |
| pyyaml | `conda install --yes pyyaml` |
| HDF5 | `conda install --yes HDF5` |
| h5py | `conda install --yes h5py` |
| gpu | `conda install --yes -c rdonnelly libgpuarray`<br> `conda install --yes -c rdonnelly pygpu`
| theano | `install --yes -c conda-forge theano=0.8*` |
| keras | `conda install --yes -c conda-forge keras=1.2*` |
| sklearn |`conda install --yes scikit-learn=0.17*`|
| rdkit | `conda install --yes -c rdkit rdkit-postgresql` |
| rpy2 | `conda install --yes -c r rpy2` |
| PRROC | `conda install --yes -c bioconda r-prroc=1.1` |
| CROC | `conda install --yes -c auto croc` |
| IRV | https://github.com/Malnammi/deepchem | 

All of the above are prerequisites. Then clone this git repo, go to home repository and setup.

```
git clone https://github.com/chao1224/virtual-screening.git
cd virtual-screening
pip install -e .
```

If a permission denied exception comes up, try to use the installment only for current user.

```
pip install --user -e .
```

## dataset

Some datasets that too big to put it here, but I upload a copy of them on the google drive, and here is the [link](https://drive.google.com/drive/folders/0B7r_bc_dhXLYLVctbC0zRnY4ZWM?usp=sharing)

keck_updated_complete.csv: contains complete data for Keck_Pria

We have a pre-fixed split dataset for Keck_Pria, PCBA, and Keck_PCBA combined.

## virtual-screening

- models:
  
  - `deep_classification.py` SingleClassification (STNN-C)
  
  - `deep_regression.py` SingleRegression (STNN-R)
  
  - `deep_classification.py` MultiClassification (MTNN-C)
  
  - `vanilla_lstm.py` VanillaLSTM (LSTM)
  
  - `sklearn_randomforest.py` RandomForest (RF)
  
  - `deepchem_irv_single.py` InfluenceRelevanceVoter (IRV)
  
- stages:

  - `stage_hyperparameter_search.py`
  
  - `stage_cross_validation.py`
  
  - `stage_prospective_screening.py`

- `data_preparation.py`

- `function.py`

- `evaluation.py`

- `integrity_checker.py`

- analysis

## json

json config files.

## test

test scripts.
