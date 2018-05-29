## analysis subdirectory

Contains scripts and notebooks to reproduce figures from the manuscript.

## models subdirectory

Contains code and instructions for training models.

## data_preparation.py and function.py

Contains functions for preparing the dataset (extraction, splitting into folds, and merging).

### Pseudocode for splitting PCBA into folds

    Algorithm:
        1- shuffle the PCBA rows randomly
        2- sort the PCBA labels from smallest active_counts to largest
        3- iterate on this sorted label list and do:
            -create k folds which will contain the row indexes only
            -split the active_indexes into the k folds
            -split the inactives_indexes into the k folds
            -split the missing_indexes into the k folds

            -take the unique compounds in each fold to remove duplicate row indexes.

            -greedily remove overlapping indexes from each fold. start with
             fold 0 and remove from the other 1-k folds. then fold 1 and remove
             from the other 2-k folds. then fold 2 and remove from the other
             3-k folds. and so on. This ensures that the top most fold contains
             the row index and all other folds do not.

        4- take the unique compounds in each fold to remove duplicate row indexes just in case

## evaluation.py

Metric function: ROC, PR, EF, NEF, etc.

## integrity_checker.py

Confirms integrity of dataset files via hashes.
