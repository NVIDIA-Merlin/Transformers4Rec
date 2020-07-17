# environment setup right after getting a ngc instance and clone recsys repo

apt-get install sshfs -y
apt-get install unzip -y
apt-get install screen -y
apt-get install nano -y

# create new virtual env and activate
conda create -n recsys python=3.7 --yes
source activate recsys

cd ~/recsys/transformers4recsys/codes

# install python packages
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --yes
pip install gpustat --yes
pip install -r requirements.txt

# get dataset
bash bash script/get_dataset.bash

# run experiment
bash script/run_transformer.bash
