The subdirectory contains examples of the expected file format.
This is the file format used for our datasets on Zenodo ([doi:10.5281/zenodo.1287964](https://doi.org/10.5281/zenodo.1287964)).
The test dataset has been split into five folds.
The columns in the csv files are:
* `Molecule`, a molecule ID, used as Index for the pandas dataframe
* `SMILES`, SMILES string representation of the compound, used for LSTM training
* `Fingerprints`, 1024 bit Morgan fingerprints representation of the compound
* `Keck_Pria_AS_Retest`, the **PriA-SSB AS** binary label
* `Keck_Pria_FP_data`, the **PriA-SSB FP** binary label
* `Keck_Pria_Continuous`, the continuous PriA-SSB primary screen % inhibition
* `Keck_RMI_cdd`, the **RMI-FANCM FP** binary label
* `FP counts % inhibition`, the continuous RMI-FANCM primary screen % inhibition

When training models for a new target, the PriA-SSB and RMI-FANCM columns can be replaced with suitable labels for that target.

The pria_lifechem subdirectory in the parent directory contains scripts for preprocessing a chemical screening dataset for training new models.

`download.sh` is a script to download the Zenodo files mentioned above and move them into the expected local directories.
