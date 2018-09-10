#!/usr/bin/env bash

model_name_list=(single_classification_22 single_classification_42 single_regression_2 single_regression_11 multi_classification_15 multi_classification_18 vanilla_lstm_8 vanilla_lstm_19)
task_list=(cross_validation_Keck_Pria_AS_Retest cross_validation_Keck_FP cross_validation_RMI)

multi_task="${array[@]}"
single_task=${array[$process]}

for model_name in "${model_name_list[@]}"
do
for task in "${task_list[@]}"
do
for fold_idx in {0..19}
do

python regenerate_results.py \
--task="$task" \
--model_name="$model_name" \
--fold_idx="$fold_idx" > ../stage_1/"$task"/"$model_name"/"$fold_idx".out

done
done
done