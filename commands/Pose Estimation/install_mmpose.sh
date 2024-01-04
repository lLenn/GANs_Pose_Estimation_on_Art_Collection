#!/bin/bash

bash ./commands/prep_environment.sh

cd lib

# Clone the mmpose API
if [ ! -d mmpose ]
then
    git clone https://github.com/lLenn/mmpose.git mmpose
    cd mmpose
else
    cd mmpose
    git pull
fi

# Setup package
python3 ./setup.py develop

cd ../..