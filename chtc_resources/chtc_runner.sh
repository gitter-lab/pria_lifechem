#!/bin/bash
mkdir job_results
mkdir job_results/deep_chem
mkdir job_results/irv
mkdir job_results/light_chem
mkdir job_results/neural_networks
mkdir job_results/random_forest

wget -q –retry-connrefused –waitretry=10 http://proxy.chtc.wisc.edu/SQUID/alnammi/Anaconda3-4.3.1-Linux-x86_64.sh #here I get the anaconda file from squid
wget -r -nH --cut-dirs=2 -np -R index.html* -q –retry-connrefused –waitretry=10 http://proxy.chtc.wisc.edu/SQUID/alnammi/dataset/ #here I get the dataset 

chmod 777 * #wget does strange things

echo 'Done getting from squid'

./Anaconda3-4.3.1-Linux-x86_64.sh -b -p ./anaconda > /dev/null #install anaconda, I also add an argument to the directory name

export PATH=$PWD/anaconda/bin:$PATH

echo 'Done installing anaconda'
chmod 777 *

#keras stuff
conda install --yes pyyaml > /dev/null
conda install --yes HDF5 > /dev/null
conda install --yes h5py > /dev/null
conda install --yes libgpuarray > /dev/null
conda install --yes -c trung pygpu > /dev/null
conda install --yes -c conda-forge theano > /dev/null
conda install --yes -c conda-forge keras=1.2.1 > /dev/null

echo 'Done installing libraries'

chmod 777 -R ./anaconda

#get virtual-screening from github
curl -H "Authorization: token 01f32242cdb9725726f581d93ef0c37e713311b7" -L https://api.github.com/repos/lscHacker/virtual-screening/zipball > virtual-screening-master.zip
unzip virtual-screening-master.zip > /dev/null

mv lsc* virtual-screening
mv ./dataset/ ./virtual-screening/dataset

cd virtual-screening

#run python job
python_jobs_dir=$1
KERAS_BACKEND=theano THEANO_FLAGS="base_compiledir=./tmp,floatX=float32,device=cuda,gpuarray.preallocate=0.8" python src/chtc_distributor.py $python_jobs_dir

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