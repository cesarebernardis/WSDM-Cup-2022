#!/bin/bash

mkdir datasets
mkdir submission

conda create -y -n wsdmcup --file requirements.txt python=3.8 -c anaconda -c conda-forge

source activate wsdmcup

#pip install -e RecSysFramework/Utils/similaripy
pip install similaripy
pip install -e RecSysFramework/Utils/pyltr
pip install implicit==0.5.2
pip install tensorflow-gpu==2.5
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
