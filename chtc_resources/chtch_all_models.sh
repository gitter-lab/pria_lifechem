#!/bin/bash
wget â€“q â€“retryâ€“connrefused â€“waitretry=10 http://proxy.chtc.wisc.edu/SQUID/alnammi/Anaconda2-4.3.1-Linux-x86_64.sh  > /dev/null 
wget -r -nH --cut-dirs=2 -np -R index.html* -q Ã¢â‚¬â€œretry-connrefused Ã¢â‚¬â€œwaitretry=10 http://proxy.chtc.wisc.edu/SQUID/alnammi/dataset/ > /dev/null 
wget -r -nH --cut-dirs=2 -np -R index.html* -q Ã¢â‚¬â€œretry-connrefused Ã¢â‚¬â€œwaitretry=10 http://proxy.chtc.wisc.edu/SQUID/alnammi/.condarc > /dev/null

chmod 777 *

echo 'Done getting from squid'

./Anaconda2-4.3.1-Linux-x86_64.sh -b -p ./anaconda > /dev/null 

export PATH=$PWD/anaconda/bin:$PATH
export HOME=$PWD

echo 'Done installing anaconda'
chmod 777 *

conda create --name rfirv
y | source activate rfirv 

#deepchem stuff
git clone https://github.com/Malnammi/deepchem.git

conda install --yes pyyaml > /dev/null
conda install --yes HDF5 > /dev/null
conda install --yes h5py > /dev/null
conda install --yes -c rdonnelly libgpuarray > /dev/null
conda install --yes -c rdonnelly pygpu > /dev/null
conda install --yes -c rdonnelly theano > /dev/null
conda install --yes -c conda-forge theano=0.8* > /dev/null
conda install --yes -c conda-forge keras=1.2* > /dev/null
conda install --yes scikit-learn=0.17* > /dev/null
conda install --yes -c rdkit rdkit-postgresql > /dev/null
conda install -y --override-channels -c conda-forge bzip2 > /dev/null
conda install -y -c conda-forge 'icu=56.*' lxml > /dev/null
conda install -y -c conda-forge 'icu=58.*' lxml > /dev/null
conda install -y -c omnia openbabel=2.4.0 > /dev/null
conda install -y -c omnia pdbfixer=1.4 > /dev/null
conda install -y -c rdkit rdkit > /dev/null
conda install -y joblib > /dev/null
conda install -y -c omnia mdtraj > /dev/null
#conda install -y scikit-learn > /dev/null
conda install -y setuptools > /dev/null
conda install -y -c conda-forge keras=1.2.2 > /dev/null
conda install -y -c conda-forge protobuf=3.1.0 > /dev/null
conda install -y -c anaconda networkx=1.11 > /dev/null
conda install -y -c bioconda xgboost=0.6a2 > /dev/null
conda install -y -c conda-forge six=1.10.0 > /dev/null
conda install -y -c conda-forge nose=1.3.7 > /dev/null
conda install --yes -c conda-forge tensorflow=1.0.0 > /dev/null
conda install --yes -c jjhelmus tensorflow-gpu=1.0.1 > /dev/null
conda install --yes mkl-service > /dev/null
conda install --yes -c conda-forge rpy2 > /dev/null
conda install --yes -c bioconda r-prroc=1.1 > /dev/null
conda install --yes -c auto croc=1.0.63 > /dev/null
conda install --yes matplotlib > /dev/null

cd deepchem
python setup.py install > /dev/null
cd ..

echo 'Done installing libraries'

chmod 777 -R ./anaconda

rm -rf ./virtual-screening/dataset
mv ./dataset/ ./virtual-screening/dataset/
export PYTHONPATH=${PYTHONPATH}:${PWD}/virtual-screening:${HOME}/virtual-screening/virtual_screening:${HOME}/virtual-screening/virtual_screening/models

cd virtual-screening

#set up matplotlib.use('Agg') so no Could not connect to display error occurs
mkdir $HOME/.config
mkdir $HOME/.config/matplotlib
echo 'backend: Agg' > $HOME/.config/matplotlib/matplotlibrc

#run python job
python_jobs_dir=$1
python virtual_screening/anaylysis/prepare_stage_1_results.py --model_dir=../job_results/ --dataset_dir=./dataset/keck/fold_5/ --output_dir=../job_results_small/

echo 'Done running job'

#THEANO_FLAGS="dnn.library_path=/usr/local/cuda-7.5/lib64,dnn.include_path=/usr/local/cuda-7.5/include,base_compiledir=./tmp,floatX=float32,device=cuda,gpuarray.preallocate=0.8" python -c 'import theano; print(theano.config)' | less

cd ..

#clean up everything I don't want transfered back
rm -f Anaconda*
rm -R -f virtual-screening*
rm -R -f deepchem*
rm -rf ./anaconda*
rm -rf ./tmp*

echo _CONDOR_JOB_IWD $_CONDOR_JOB_IWD
echo Cluster $cluster
echo Process $process
echo RunningOn $runningon
