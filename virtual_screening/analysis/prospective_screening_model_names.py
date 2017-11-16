model_name_mapping = {
    'baseline': 'Baseline',
    
    # dnn
    'single_regression_2': 'SingleRegression_a',
    'single_regression_11': 'SingleRegression_b',
    'single_classification_22': 'SingleClassification_a',
    'single_classification_42': 'SingleClassification_b',
    'multi_classification_15': 'MultiClassification_a',
    'multi_classification_18': 'MultiClassification_b',
    'vanilla_lstm_8': 'LSTM_a',
    'vanilla_lstm_19': 'LSTM_b',

    # random forest
    'random_forest_12': 'RandomForest_a',
    'random_forest_13': 'RandomForest_b',
    'random_forest_14': 'RandomForest_c',
    'random_forest_24': 'RandomForest_d',
    'random_forest_25': 'RandomForest_e',
    'random_forest_72': 'RandomForest_f',
    'random_forest_96': 'RandomForest_g',
    'random_forest_97': 'RandomForest_h',

    # irv
    'irv_5': 'IRV_a',
    'irv_10': 'IRV_b',
    'irv_20': 'IRV_c',
    'irv_40': 'IRV_d',
    'irv_80': 'IRV_e',

    # docking
    'dockscore_hybrid': 'Docking_hybrid',
    'dockscore_fred': 'Docking_fred',
    'dockscore_dock6': 'Docking_dock6',
    'dockscore_rdockint': 'Docking_rdockint',
    'dockscore_rdocktot': 'Docking_rdocktot',
    'dockscore_surflex': 'Docking_surflex',
    'dockscore_ad4': 'Docking_ad4',
    'dockscore_plants': 'Docking_plants',
    'dockscore_smina': 'Docking_smina',
    'consensus_dockscore_max': 'ConsensusDocking_max',
    'consensus_bcs_efr1_opt': 'ConsensusDocking_efr1_opt',
    'consensus_bcs_rocauc_opt': 'ConsensusDocking_rocauc_opt',
    'consensus_dockscore_median': 'ConsensusDocking_median',
    'consensus_dockscore_mean': 'ConsensusDocking_mean',

    # lightchemPRAUC_1Layer1Model
    'lightchem_PRAUC_1Layer1Model_test_lc4': 'CBF_a',
    'lightchem_PRAUC_5Layer1Model_test_lc4': 'CBF_b',
    'lightchem_PRAUC_10Layer1Model_test_lc4': 'CBF_c',
    'lightchem_ROCAUC_1Layer1Model_test_lc4': 'CBF_d',
    'lightchem_ROCAUC_5Layer1Model_test_lc4': 'CBF_e',
    'lightchem_ROCAUC_10Layer1Model_test_lc4': 'CBF_f',
}