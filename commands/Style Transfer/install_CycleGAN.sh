#!/bin/bash

bash ./commands/prep_environment.sh

cd lib

# Clone the CycleGAN API
if [ ! -d CycleGAN ]
then
    git clone https://github.com/lLenn/pytorch-CycleGAN-and-pix2pix.git CycleGAN
    cd CycleGAN
else
    cd CycleGAN
    git pull
fi

if [ ! -f "__init__.py" ]
then
    touch "__init__.py"
fi

# Setup package
cd ..
cp ../commands/Style\ Transfer/setup.py ./setup.py
python3 ./setup.py develop
cd CycleGAN

# Install dependencies
pip3 install -r requirements.txt

cd ../..