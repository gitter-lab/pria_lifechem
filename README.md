# virtual-screening

## dataset

Some datasets that too big to put it here, but I upload a copy of them on the google drive, and here is the [link](https://drive.google.com/drive/folders/0B7r_bc_dhXLYLVctbC0zRnY4ZWM?usp=sharing)

keck_updated_complete.csv: contains complete data for Keck_Pria

'5_fold_split_data', '10_fold_split_data': two directories for fixed data split for training and testing. Please refer to src/demo.ipynb for more details.

lc123_keckdata.sdf: used for data preparation

## src

data_preparation.py: fixed-data inside, combine all the updated data into one csv file

function.py: all helper functions

evaluation.py: all evaluation functions

demo.ipynb gives some examples on how to use them.

## setup

`sudo pip install -e .`

if permission denied, try

`pip install --user -e .`