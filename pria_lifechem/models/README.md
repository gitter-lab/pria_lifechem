## Models

| Model | script name |
| --- | --- |
| Random Forest | `sklearn_randomforest.py` |
| IRV | `deepchem_irv.py` |
| Single-Task Neural Network Classification | `deep_classification.py` |
| Single-Task Neural Network Regression | `deep_regression.py` |
| Multi-Task Neural Network Classification | `deep_classification.py` |
| LSTM-Network | `vanilla_lstm.py` |
| Tree-Net | `tree_net.py` |

Note that the `tree_net.py` models were not used in our manuscript.

## Hyperparameter Stage

**Running CMD for random forest**

```
python sklearn_randomforest.py \
--config_json_file=../../json/sklearn_randomforest.json \
--model_dir=$results_and_model_directory \
--dataset_dir=$path_to_dataset \
--process_num=$process \
--stage=0
```

There are a total of `108` hyperparameter combinations.
Each `process_num` from `0:107` will run one of them.
Results and models will be saved to `model_dir`.

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

**Running CMD for random forest**

```
python sklearn_randomforest.py \
--config_json_file=../../json/sklearn_randomforest.json \
--model_dir=$results_and_model_directory \
--dataset_dir=$path_to_dataset \
--process_num=$process \
--stage=1
```

There are a total of `108` hyperparameter combinations.
`8` random forests were promoted to the cross-validation stage.

The following are the corresponding `process_num` for the parameters: `[12, 13, 14, 24, 25, 72, 96, 97]`

**Running CMD for IRV**

```
knn=(5 10 20 40 80)
i=$(echo $(( $process / 15 )))
k=${knn[$i]}

python deepchem_irv.py \
--config_json_file=../../json/deepchem_irv.json \
--model_dir=../job_results/irv/deepchem_irv_${k}/ \
--dataset_dir=$path_to_dataset \
--process_num=$process \
--stage=1
```

We run IRV for each fold (5-total) and label (3-total) as a single process.
This allows parallel training of folds.
This can be seen in the `json/deepchem_irv.json` file.
So, for each `knn`, we run 15 processes.

Thus, for the cross-validation stage, run `process_num` from `0:74`.

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

**Running CMD for random forest**

```
python sklearn_randomforest.py \
--config_json_file=../../json/sklearn_randomforest.json \
--model_dir=$results_and_model_directory \
--dataset_dir=$path_to_dataset \
--process_num=$process \
--stage=2
```

Run for `process_num`: `[12, 13, 14, 24, 25, 72, 96, 97]`

**Running CMD for IRV**

```
knn=(5 10 20 40 80)
i=$(echo $(( $process / 3 )))
k=${knn[$i]}

python deepchem_irv.py \
--config_json_file=../../json/deepchem_irv.json \
--model_dir=../job_results/irv/deepchem_irv_${k}/ \
--dataset_dir=$path_to_dataset \
--process_num=$process \
--stage=2
```

We run IRV for each fold (1-total) and label (3-total) as a single process.

Thus, for the prospective screening stage, run `process_num` from `0:14`.

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

`CallBacks.py`: Keras callback functions for early stopping.
Used in neural network models.

## Tester

`test_demo.py` is used for a quick test on sampled dataset.