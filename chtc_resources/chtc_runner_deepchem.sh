#!/bin/bash
mkdir job_results
mkdir job_results/deep_chem
mkdir job_results/irv
mkdir job_results/light_chem
mkdir job_results/neural_networks
mkdir job_results/random_forest

wget –q –retry–connrefused –waitretry=10 http://proxy.chtc.wisc.edu/SQUID/alnammi/Anaconda2-4.3.1-Linux-x86_64.sh #here I get the anaconda file from squid
wget -r -nH --cut-dirs=2 -np -R index.html* -q â€“retry-connrefused â€“waitretry=10 http://proxy.chtc.wisc.edu/SQUID/alnammi/dataset/ #here I get the dataset 

chmod 777 * #wget does strange things

echo 'Done getting from squid'

./Anaconda2-4.3.1-Linux-x86_64.sh -b -p ./anaconda > /dev/null #install anaconda, I also add an argument to the directory name

export PATH=$PWD/anaconda/bin:$PATH
export HOME=$_CONDOR_JOB_IWD

echo 'Done installing anaconda'
chmod 777 *

#deepchem stuff
git clone https://github.com/Malnammi/deepchem.git

conda install -y -c omnia openbabel=2.4.0 > /dev/null
conda install -y -c omnia pdbfixer=1.4 > /dev/null
conda install -y -c rdkit rdkit > /dev/null
conda install -y joblib > /dev/null
conda install -y -c omnia mdtraj > /dev/null
conda install -y scikit-learn > /dev/null
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

#wget -q –retry-connrefused –waitretry=10 http://github.com/deepchem/deepchem/archive/master.zip > /dev/null
#unzip master.zip > /dev/null
#rm -f master.zip
#mv deepchem-master* deepchem

cd deepchem
python setup.py install
nosetests -v deepchem --nologcapture
source activate deepchem
cd ..

echo 'Done installing libraries'

chmod 777 -R ./anaconda

#get virtual-screening from github
curl -H "Authorization: token 5879e760ce2f7b753aa80bda34811162ec7ababe" -L https://api.github.com/repos/lscHacker/virtual-screening/zipball > virtual-screening-master.zip
unzip virtual-screening-master.zip > /dev/null
rm -f virtual-screening-master.zip

mv lsc* virtual-screening
rm -rf ./virtual-screening/dataset
mv ./dataset/ ./virtual-screening/dataset/
export PYTHONPATH=${PYTHONPATH}:${_CONDOR_JOB_IWD}/virtual-screening:${_CONDOR_JOB_IWD}/virtual-screening/virtual_screening:${_CONDOR_JOB_IWD}/virtual-screening/virtual_screening/models

cd virtual-screening

#set up matplotlib.use('Agg') so no Could not connect to display error occurs
mkdir $HOME/.config
mkdir $HOME/.config/matplotlib
echo 'backend: Agg' > $HOME/.config/matplotlib/matplotlibrc

#run python job
python_jobs_dir=$1
python virtual_screening/models/deepchem_irv.py --config_json_file=./json/deepchem_irv.json --model_dir=../job_results/irv/deepchem_irv_${cluster}_${process}/ --dataset_dir=./dataset/keck/fold_5/

echo 'Done running job'

#THEANO_FLAGS="dnn.library_path=/usr/local/cuda-7.5/lib64,dnn.include_path=/usr/local/cuda-7.5/include,base_compiledir=./tmp,floatX=float32,device=cuda,gpuarray.preallocate=0.8" python -c 'import theano; print(theano.config)' | less

cd ..

#clean up everything I don't want transfered back
rm -f Anaconda*
rm -R -f virtual-screening*
rm -R -f deepchem*
rm -rf ./anaconda*


echo _CONDOR_JOB_IWD $_CONDOR_JOB_IWD
echo Cluster $cluster
echo Process $process
echo RunningOn $runningon