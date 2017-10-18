"""
dictionary mapping for model names
"""
def model_name_dict():
    return {
    #lightchem
    "PRAUC_5Layer1Model" : "LightChem_PRAUC_5Layer1",
    "PRAUC_10Layer1Model"  : "LightChem_PRAUC_10Layer1",
    "ROCAUC_5Layer1Model" : "LightChem_ROCAUC_5Layer1",
    "ROCAUC_10Layer1Model" : "LightChem_ROCAUC_10Layer1",
    "ROCAUC_1Layer1Model" : "LightChem_ROCAUC_1Layer1",
    "PRAUC_1Layer1Model" : "LightChem_PRAUC_1Layer1",
    #random forest
    "sklearn_rf_390014_24" : "sklearn_rf_24",
    "sklearn_rf_390014_25" : "sklearn_rf_25",
    "sklearn_rf_390014_97" : "sklearn_rf_97",
    "sklearn_rf_390014_96" : "sklearn_rf_96",
    "sklearn_rf_390014_14" : "sklearn_rf_14",
    "sklearn_rf_390014_12" : "sklearn_rf_12",
    "sklearn_rf_390014_13" : "sklearn_rf_13",
    "sklearn_rf_390014_72" : "sklearn_rf_72",
    #dnn
    "single_regression_11" : "single_regression_11",
    "multi_classification_18" : "multi_classification_18",
    "multi_classification_15" : "multi_classification_15",
    "single_regression_2" : "single_regression_2",
    "single_classification_42" : "single_classification_42",
    "single_classification_22" : "single_classification_22",
    "vanilla_lstm_8" : "vanilla_lstm_8",
    "vanilla_lstm_19" : "vanilla_lstm_19",
    #irv
    "deepchem_irv_5" : "deepchem_irv_5",
    "deepchem_irv_10" : "deepchem_irv_10",
    "deepchem_irv_20" : "deepchem_irv_20",
    "deepchem_irv_40" : "deepchem_irv_40",
    "deepchem_irv_80" : "deepchem_irv_80",
    #docking
    "dockscore_hybrid" : "dockscore_hybrid",   
    "dockscore_fred" : "dockscore_fred",
    "dockscore_dock6" : "dockscore_dock6",
    "dockscore_rdockint" : "dockscore_rdockint",
    "dockscore_rdocktot" : "dockscore_rdocktot",
    "consensus_dockscore_max" : "consensus_dockscore_max",
    "consensus_bcs_efr1_opt" : "consensus_bcs_efr1_opt",
    "consensus_bcs_rocauc_opt" : "consensus_bcs_rocauc_opt",
    "consensus_dockscore_median" : "consensus_dockscore_median",
    "dockscore_surflex" : "dockscore_surflex",
    "dockscore_ad4" : "dockscore_ad4",
    "consensus_dockscore_mean" : "consensus_dockscore_mean",
    "dockscore_plants" : "dockscore_plants",
    "dockscore_smina" : "dockscore_smina",
    }
