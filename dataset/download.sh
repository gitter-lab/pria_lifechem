#!/usr/bin/env bash
wget https://zenodo.org/record/1287964/files/pria_rmi_cv.tar.gz
wget https://zenodo.org/record/1287964/files/pria_rmi_pcba_cv.tar.gz
wget https://zenodo.org/record/1287964/files/pria_prospective.csv.gz

tar -xzvf pria_rmi_cv.tar.gz
tar -xzvf pria_rmi_pcba_cv.tar.gz

mkdir -p fixed_dataset/fold_5
mkdir -p keck_pcba/fold_5

mv pria_rmi_cv/* fixed_dataset/fold_5
mv pria_rmi_pcba_cv/* keck_pcba/fold_5

rm pria_rmi_cv.tar.gz
rm pria_rmi_pcba_cv.tar.gz

