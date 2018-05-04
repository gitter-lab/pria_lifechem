## Models

| Model | script name |
| --- | --- |
| Random Forest | `sklearn_randomforest.py` |
| IRV | `deepchem_irv_single.py` |
| Single-Task Neural Network Classification | `deep_classification.py` |
| Single-Task Neural Network Regression | `deep_classification.py` |
| Multi-Task Neural Network Classification | `deep_classification.py` |
| LSTM-Network | `vanilla_lstm.py` |
| Tree-Net | `tree_net.py` |

## Helper Files

+ `CallBacks.py`: keras callback functions for early stopping. Used in neural network models.

+ `cross_validation.py`: General API to run neural network and random forest models.

## Note
All imeplementations are intended to run on [HTCondor](http://research.cs.wisc.edu/htcondor/manual/). 
Jobs are submitted and each job is identified by a cluster and process id. 
Extract the relevant model architecture code to run on a local machine.
