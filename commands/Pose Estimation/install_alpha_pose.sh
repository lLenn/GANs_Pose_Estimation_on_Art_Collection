#!/bin/bash

# Create tmp folder for install
mkdir lib
cd lib

# Downgrade pip to avoid build error
pip install pip==21.2

# Install PyTorch
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
if [[ ${CUDA_VERSION} == 9.0* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;7.0+PTX"
elif [[ ${CUDA_VERSION} == 9.2* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0+PTX"
elif [[ ${CUDA_VERSION} == 10.* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5+PTX"
elif [[ ${CUDA_VERSION} == 11.0* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0+PTX"
elif [[ ${CUDA_VERSION} == 11.* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
else
    echo "unsupported cuda version."
    exit 1
fi

# install
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
pip install cython pycocotools
sudo apt-get install libyaml-dev

git clone https://github.com/HaoyiZhu/HalpeCOCOAPI.git
cd HalpeCOCOAPI/PythonAPI
python3 setup.py build develop --user

cd ../..
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose
python3 setup.py build develop

# update pip
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py

#remove lib
#cd ../..
#rm -rf lib