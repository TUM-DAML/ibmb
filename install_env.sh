#!/bin/bash

# new env
echo y | conda create -n LBMB python=3.7
source /nfs/students/qian/miniconda3/etc/profile.d/conda.sh
conda activate LBMB

# pytorch + cudatoolkit
echo y | conda install pytorch==1.8.1 cudatoolkit=11.1 -c pytorch -c conda-forge


# PyG
export CUDA=111
export TORCH=1.8.1

# install from https://pytorch-geometric.com/whl/
# install local wheel!
pip install torch-geometric==1.7.0

# jupyter related
pip install jupyter
pip install jupyter-resource-usage

# others
pip install tensorboard
pip install seml
pip install numba==0.53.1
pip install python-tsp
pip install ogb
pip install parallel-sort
pip install joblib