#!/bin/bash

# Update pip
python.exe -m pip install --upgrade pip

# Create tmp folder for install
if [ ! -d bin ]
then
    mkdir bin
fi
cd bin

# Clone the COCO API
if [ ! -d cocoapi ]
then
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python.exe setup.py install
    cd ../..
fi

# Clone the CrowdPose API
if [ ! -d CrowdPose ]
then
    git clone https://github.com/Jeff-sjtu/CrowdPose.git
    cd CrowdPose/crowdpose-api/PythonAPI
    python.exe setup.py install
    cd ../../..
fi

# Clone the fork of SWAHR-HumanPose
if [ ! -d SWAHR-HumanPose ]
then
    git clone https://github.com/lLenn/SWAHR-HumanPose.git
fi
cd SWAHR-HumanPose

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
pip install gdown
if [ ! -d models ]
then
    mkdir models
    gdown --folder --id 1LTC9BqodDw3qfQ2DjgH0f_n3Javds0Pe -O models/pose_coco
    gdown --folder --id 1-W9OoshMaT5UvBaW8vPhpP8pEphIcKJ7 -O models/pose_crowdpose
fi

cd ../..