#!/bin/bash
mkdir job_results
mkdir job_results/deep_chem
mkdir job_results/irv
mkdir job_results/light_chem
mkdir job_results/neural_networks
mkdir job_results/random_forest

wget -q –retry-connrefused –waitretry=10 https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh #here I get the anaconda file from squid
wget -r -nH --cut-dirs=2 -np -R index.html* -q â€“retry-connrefused â€“waitretry=10 http://proxy.chtc.wisc.edu/SQUID/alnammi/dataset/ #here I get the dataset 

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
conda install --yes -c rdonnelly libgpuarray > /dev/null
conda install --yes -c rdonnelly pygpu > /dev/null
conda install --yes -c rdonnelly theano=0.8* > /dev/null
conda install --yes -c conda-forge keras=1.2* > /dev/null
conda install --yes scikit-learn=0.17* > /dev/null
conda install --yes pandas > /dev/null
conda install --yes -c rdkit rdkit-postgresql > /dev/null
conda install --yes -c r rpy2 > /dev/null
conda install --yes -c bioconda r-prroc=1.1 > /dev/null
conda install --yes -c auto croc=1.0.63 > /dev/null

echo 'Done installing libraries'

chmod 777 -R ./anaconda

#get virtual-screening from github
curl -H "Authorization: token 5879e760ce2f7b753aa80bda34811162ec7ababe" -L https://api.github.com/repos/chao1224/virtual-screening/zipball > virtual-screening-master.zip
unzip virtual-screening-master.zip > /dev/null
rm -f virtual-screening-master.zip

mv chao* virtual-screening
rm -rf ./virtual-screening/dataset
mv ./dataset/ ./virtual-screening/dataset/
export PYTHONPATH=${PYTHONPATH}:${_CONDOR_JOB_IWD}/virtual-screening:${_CONDOR_JOB_IWD}/virtual-screening/src:${_CONDOR_JOB_IWD}/virtual-screening/src/models

cd virtual-screening


#run python job
python_jobs_dir=$1
KERAS_BACKEND=theano THEANO_FLAGS="base_compiledir=./tmp,floatX=float32,device=gpu,gpuarray.preallocate=0.8" python src/chtc_distributor.py $python_jobs_dir

echo 'Done running job'

#THEANO_FLAGS="dnn.library_path=/usr/local/cuda-7.5/lib64,dnn.include_path=/usr/local/cuda-7.5/include,base_compiledir=./tmp,floatX=float32,device=cuda,gpuarray.preallocate=0.8" python -c 'import theano; print(theano.config)' | less

cd ..

#clean up everything I don't want transfered back
rm -f Anaconda*
rm -R -f virtual-screening*
rm -rf ./anaconda*


echo _CONDOR_JOB_IWD $_CONDOR_JOB_IWD
echo Cluster $cluster
echo Process $process
echo RunningOn $runningon
