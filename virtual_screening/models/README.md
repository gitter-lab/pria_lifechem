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

## Hyperparameter Stage

**Running CMD for single-task classification**

```
process=0
$transfer_output_files=/path/to/output/directory

KERAS_BACKEND=theano \
THEANO_FLAGS="base_compiledir=./tmp,device=gpu,floatX=float32,gpuarray.preallocate=0.8" \
python stage_hyperparameter_search.py \
--config_json_file=../../json/single_classification.json \
--PMTNN_weight_file=$transfer_output_files/$process.weight \
--config_csv_file=$transfer_output_files/$process.result.csv \
--process_num=$process \
--model=single_classification > $transfer_output_files/$process.out
```

## Cross-Validation Stage

**Running CMD for single-task classification**

```
process=0
$transfer_output_files=/path/to/output/directory

KERAS_BACKEND=theano \
THEANO_FLAGS="base_compiledir=./tmp,device=gpu,floatX=float32,gpuarray.preallocate=0.8" \
python stage_cross_validation.py \
--config_json_file=../../json/single_classification.json \
--PMTNN_weight_file=$transfer_output_files/$process.weight \
--config_csv_file=$transfer_output_files/$process.result.csv \
--process_num=$process \
--model=single_classification > $transfer_output_files/$process.out
```

## Prospective-Screening Stage

**Running CMD for single-task classification**

```
process=0
$transfer_output_files=/path/to/output/directory

KERAS_BACKEND=theano \
THEANO_FLAGS="base_compiledir=./tmp,device=gpu,floatX=float32,gpuarray.preallocate=0.8" \
python stage_prospective_screening.py \
--config_json_file=../../json/single_classification.json \
--PMTNN_weight_file=$transfer_output_files/$process.weight \
--config_csv_file=$transfer_output_files/$process.result.csv \
--model=single_classification > $transfer_output_files/$process.out
```

## Helper Files

`CallBacks.py`: keras callback functions for early stopping. Used in neural network models.

## Note
All imeplementations are intended to run on [HTCondor](http://research.cs.wisc.edu/htcondor/manual/). 
Jobs are submitted and each job is identified by a cluster and process id. 
Extract the relevant model architecture code to run on a local machine.
