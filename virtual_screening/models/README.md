## Models

| Random Forest | `sklearn_randomforest.py` |
| IRV | `deepchem_irv_single.py` |
| Single-Task Neural Network Classification | `deep_classification.py` |
| Single-Task Neural Network Regression | `deep_classification.py` |
| Multi-Task Neural Network Classification | `deep_classification.py` |
| LSTM-Network | `vanilla_lstm.py` |

## Helper Files

`CallBacks.py`: keras callback functions for early stopping. Used in neural network models.

`cross_validation.py`: Helper class for running neural network models in the cross-validation stage.

## Note
All imeplementations are intended to run on [HTCondor](http://research.cs.wisc.edu/htcondor/manual/). 
Jobs are submitted and each job is identified by a cluster and process id. 
Extract the relevant model architecture code to run on a local machine.
