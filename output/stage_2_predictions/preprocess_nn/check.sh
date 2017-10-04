#!/bin/bash

date

task_list=(Keck_Pria_AS_Retest Keck_FP RMI)
task_list=(Keck_Pria_AS_Retest)
for task in "${task_list[@]}"
do
    output_file=../stage_2/"$task"

    ########################### single classification ###########################
    model=single_classification

    model_n="$model"_22
    echo Start
    echo $model_n
    python check_model.py \
    --config_json_file=../"$output_file"/"$model_n".json \
    --PMTNN_weight_file=../"$output_file"/"$model_n"/"$model_n".weight \
    --storage_file=../"$task"/"$model_n".npz \
    --model="$model"

    model_n="$model"_42
    echo Start
    echo $model_n
    python check_model.py \
    --config_json_file=../"$output_file"/"$model_n".json \
    --PMTNN_weight_file=../"$output_file"/"$model_n"/"$model_n".weight \
    --storage_file=../"$task"/"$model_n".npz \
    --model="$model"

    ########################### single regression ###########################
    model=single_regression

    model_n="$model"_2
    echo Start
    echo $model_n
    python check_model.py \
    --config_json_file=../"$output_file"/"$model_n".json \
    --PMTNN_weight_file=../"$output_file"/"$model_n"/"$model_n".weight \
    --storage_file=../"$task"/"$model_n".npz \
    --model="$model"

    model_n="$model"_11
    echo Start
    echo $model_n
    python check_model.py \
    --config_json_file=../"$output_file"/"$model_n".json \
    --PMTNN_weight_file=../"$output_file"/"$model_n"/"$model_n".weight \
    --storage_file=../"$task"/"$model_n".npz \
    --model="$model"

    ########################### vanilla lstm ###########################
    model=vanilla_lstm

    model_n="$model"_8
    echo Start
    echo $model_n
    python check_model.py \
    --config_json_file=../"$output_file"/"$model_n".json \
    --PMTNN_weight_file=../"$output_file"/"$model_n"/"$model_n".weight \
    --storage_file=../"$task"/"$model_n".npz \
    --model="$model"

    model_n="$model"_19
    echo Start
    echo $model_n
    python check_model.py \
    --config_json_file=../"$output_file"/"$model_n".json \
    --PMTNN_weight_file=../"$output_file"/"$model_n"/"$model_n".weight \
    --storage_file=../"$task"/"$model_n".npz \
    --model="$model"

    ########################## multi classification ###########################
    model=multi_classification

    model_n="$model"_15
    echo Start
    echo $model_n
    python check_model.py \
    --config_json_file=../"$output_file"/"$model_n".json \
    --PMTNN_weight_file=../"$output_file"/"$model_n"/"$model_n".weight \
    --storage_file=../"$task"/"$model_n".npz \
    --model="$model"

    model_n="$model"_18
    echo Start
    echo $model_n
    python check_model.py \
    --config_json_file=../"$output_file"/"$model_n".json \
    --PMTNN_weight_file=../"$output_file"/"$model_n"/"$model_n".weight \
    --storage_file=../"$task"/"$model_n".npz \
    --model="$model"
done

date