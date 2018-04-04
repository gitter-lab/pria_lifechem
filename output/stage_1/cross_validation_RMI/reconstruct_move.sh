#!/usr/bin/env bash

task=Keck_RMI_cdd

rm -rf random_forest*
rm -rf irv*

cp -r ../../../job_results_small/random_forest/stage_1/sklearn_rf_390014_12/alt_format/"$task" ./random_forest_12
cp -r ../../../job_results_small/random_forest/stage_1/sklearn_rf_390014_13/alt_format/"$task"  ./random_forest_13
cp -r ../../../job_results_small/random_forest/stage_1/sklearn_rf_390014_14/alt_format/"$task"  ./random_forest_14
cp -r ../../../job_results_small/random_forest/stage_1/sklearn_rf_390014_24/alt_format/"$task"  ./random_forest_24
cp -r ../../../job_results_small/random_forest/stage_1/sklearn_rf_390014_25/alt_format/"$task"  ./random_forest_25
cp -r ../../../job_results_small/random_forest/stage_1/sklearn_rf_390014_72/alt_format/"$task"  ./random_forest_72
cp -r ../../../job_results_small/random_forest/stage_1/sklearn_rf_390014_96/alt_format/"$task"  ./random_forest_96
cp -r ../../../job_results_small/random_forest/stage_1/sklearn_rf_390014_97/alt_format/"$task"  ./random_forest_97

cp -r ../../../job_results_small/irv/stage_1/deepchem_irv_5/alt_format/"$task"/ ./irv_5
cp -r ../../../job_results_small/irv/stage_1/deepchem_irv_10/alt_format/"$task"/ ./irv_10
cp -r ../../../job_results_small/irv/stage_1/deepchem_irv_20/alt_format/"$task"/ ./irv_20
cp -r ../../../job_results_small/irv/stage_1/deepchem_irv_40/alt_format/"$task"/ ./irv_40
cp -r ../../../job_results_small/irv/stage_1/deepchem_irv_80/alt_format/"$task"/ ./irv_80

dock_list=(dockscore_hybrid dockscore_fred dockscore_dock6 dockscore_rdockint dockscore_rdocktot dockscore_surflex
dockscore_ad4 dockscore_plants dockscore_smina consensus_dockscore_max consensus_bcs_efr1_opt consensus_bcs_rocauc_opt
consensus_dockscore_median consensus_dockscore_mean)

rm -rf consensus_*
rm -rf dockscore_*

for docking_program in "${dock_list[@]}"; do
    cp -r ../../../job_results_small/docking/stage_1/"$docking_program"/alt_format/"$task"/ ./$docking_program
done