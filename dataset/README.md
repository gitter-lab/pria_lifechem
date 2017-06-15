Under this folder, we have two folders and one cvs file.

1. Folder data_preprocessing: contains the required datasets we need for preprocessing, the function is under [src/data_preparation.py](https://github.com/lscHacker/virtual-screening/blob/master/src/data_preparation.py)

2. Folder fixed_dataset: folders with different k, splitting function is under [src/function.py](https://github.com/lscHacker/virtual-screening/blob/master/src/function.py), split [keck_original_complete.csv]() into k folders. And merging function is in [src/function.py](https://github.com/lscHacker/virtual-screening/blob/master/src/function.py).

3. File keck_original_complete.csv: the output of data_preprocessing. Contains all the data.
    * Molecule, molecule ID, used as Index for Python pandas
    * SMILES, SMILES feature, used for LSTM
    * Fingerprints, 1024 FPs, used for canonical NN feature
    * Keck_Pria_AS_Retest, 1st Pria binary label
    * Keck_Pria_FP_data, 2nd Pria binary label
    * Keck_Pria_Hard_Thresholded, 3rd Pria binary label, hard thresholded on 35
    * Keck_Pria_Continuous, Pria continuous label
    * Keck_RMI_cdd, RMI binary label
    * FP counts % inhibition, RMI continuous label
