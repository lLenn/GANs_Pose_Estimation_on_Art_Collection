#!/bin/bash

# Update pip
echo "Update pip"
python3 -m pip install --upgrade pip

# Install PyTorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Create tmp folder for install
if [ ! -d lib ]
then
    mkdir lib
fi