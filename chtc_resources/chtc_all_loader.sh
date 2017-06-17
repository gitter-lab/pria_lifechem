#!/bin/bash
wget –q –retry–connrefused –waitretry=10 http://proxy.chtc.wisc.edu/SQUID/alnammi/Anaconda2-4.3.1-Linux-x86_64.sh #here I get the anaconda file from squid
wget -r -nH --cut-dirs=2 -np -R index.html* -q â€“retry-connrefused â€“waitretry=10 http://proxy.chtc.wisc.edu/SQUID/alnammi/dataset/ #here I get the dataset 

chmod 777 * #wget does strange things

echo 'Done getting from squid'

./Anaconda2-4.3.1-Linux-x86_64.sh -b -p ./anaconda > /dev/null #install anaconda, I also add an argument to the directory name

export PATH=$PWD/anaconda/bin:$PATH
export HOME=$PWD

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
conda install --yes -c r rpy2 > /dev/null
conda install --yes -c bioconda r-prroc=1.1 > /dev/null
conda install --yes -c auto croc=1.0.63 > /dev/null

#wget -q –retry-connrefused –waitretry=10 http://github.com/deepchem/deepchem/archive/master.zip > /dev/null
#unzip master.zip > /dev/null
#rm -f master.zip
#mv deepchem-master* deepchem

cd deepchem
python setup.py install
cd ..

echo 'Done installing libraries'

chmod 777 -R ./anaconda

#get virtual-screening from github
curl -H "Authorization: token 5879e760ce2f7b753aa80bda34811162ec7ababe" -L https://api.github.com/repos/lscHacker/virtual-screening/zipball > virtual-screening-master.zip
unzip virtual-screening-master.zip > /dev/null
rm -f virtual-screening-master.zip

mv chao* virtual-screening
rm -rf ./virtual-screening/dataset
mv ./dataset/ ./virtual-screening/dataset/
export PYTHONPATH=${PYTHONPATH}:${PWD}/virtual-screening:${HOME}/virtual-screening/virtual_screening:${HOME}/virtual-screening/virtual_screening/models

cd virtual-screening/analysis


#run python job
python all_loader_tester.py

echo 'Done running job'