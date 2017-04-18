#!/bin/bash

echo Cluster $cluster
echo Process $process
echo RunningOn $runningon
mkdir $transfer_output_files
transfer_output_files=$transfer_output_files/$cluster
echo $transfer_output_files
mkdir $transfer_output_files

echo _CONDOR_JOB_IWD $_CONDOR_JOB_IWD
echo Cluster $cluster
echo Process $process
echo RunningOn $runningon

wget -q –retry-connrefused –waitretry=10 https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh #here I get the anaconda file from squid

chmod 777 * #wget does strange things

echo 'Done getting from squid'

./Anaconda2-4.3.1-Linux-x86_64.sh -b -p ./anaconda > /dev/null #install anaconda, I also add an argument to the directory name

export PATH=$PWD/anaconda/bin:$PATH

echo 'Done installing anaconda'
chmod 777 *

#keras stuff
conda install --yes pyyaml > /dev/null
conda install --yes HDF5 > /dev/null
conda install --yes h5py > /dev/null
conda install --yes libgpuarray > /dev/null
conda install --yes -c trung pygpu > /dev/null
conda install --yes pandas > /dev/null
conda install --yes -c conda-forge theano > /dev/null
conda install --yes -c conda-forge keras=1.2* > /dev/null
conda install --yes scikit-learn=0.17* > /dev/null
conda install --yes -c rdkit rdkit-postgresql > /dev/null
conda install --yes -c r rpy2 > /dev/null
conda install --yes -c bioconda r-prroc=1.1 > /dev/null
conda install --yes -c auto croc > /dev/null

echo 'Done installing libraries'

chmod 777 -R ./anaconda

#get virtual-screening from github
curl -H "Authorization: token 01f32242cdb9725726f581d93ef0c37e713311b7" -L https://api.github.com/repos/lscHacker/virtual-screening/zipball > virtual-screening-master.zip
unzip virtual-screening-master.zip > /dev/null
mv lsc* virtual-screening
cp -r dataset/* virtual-screening/dataset/

echo
#run python job
cd virtual-screening/virtual_screening/models
echo "start"
date
KERAS_BACKEND=theano \
THEANO_FLAGS="base_compiledir=./tmp,floatX=float32,gpuarray.preallocate=0.8" \
python grid_search_optimization.py \
--config_json_file=../../json/single_classification.json \
--PMTNN_weight_file=$_CONDOR_JOB_IWD/$transfer_output_files/$process.weight \
--config_csv_file=$_CONDOR_JOB_IWD/$transfer_output_files/$process.out.csv \
--SMILES_mapping_json_file=../../json/SMILES_mapping.json \
--process_num=$process --model=single_classification

echo 'Done running job'
date
