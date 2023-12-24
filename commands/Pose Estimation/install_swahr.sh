#!/bin/bash

bash ./commands/prep_environment.sh

cd lib

# Clone the COCO API
if [ ! -d cocoapi ]
then
    git clone https://github.com/lLenn/cocoapi.git
    cd cocoapi/PythonAPI
else
    cd cocoapi
    git pull
    cd PythonAPI
fi

python3 setup.py build install
cd ../..

# Clone the CrowdPose API
if [ ! -d CrowdPose ]
then
    git clone https://github.com/lLenn/CrowdPose.git
    cd CrowdPose/crowdpose-api/PythonAPI
else
    cd CrowdPose
    git pull
    cd crowdpose-api/PythonAPI
fi

python3 setup.py build install
cd ../../..

# Clone the fork of SWAHR-HumanPose
if [ ! -d SWAHR ]
then
    git clone https://github.com/lLenn/SWAHR-HumanPose.git SWAHR
    cd SWAHR
else
    cd SWAHR
    git pull
fi

# Setup package
cd lib
python3 ./setup.py develop
cd ../../..