#!/bin/bash

bash ./commands/prep_environment.sh

cd lib

# Clone the CycleGAN API
if [ ! -d CycleGAN ]
then
    git clone https://github.com/lLenn/pytorch-CycleGAN-and-pix2pix.git CycleGAN
    cd CycleGAN
    bash ./scripts/download_cyclegan_model.sh style_cezanne
else
    cd CycleGAN
fi

# Setup package
cd ..
cp ../commands/Style\ Transfer/setup.py ./setup_cyclegan.py
python3 ./setup_cyclegan.py develop
cd CycleGAN

# Install dependencies
pip3 install -r requirements.txt

cd ../..