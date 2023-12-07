#!/bin/bash

bash ./commands/prep_environment.sh

cd lib

# Clone the cirtorch API
if [ ! -d cirtorch ]
then
    git clone https://github.com/lLenn/cnnimageretrieval-pytorch.git cirtorch
    cd cirtorch
else
    cd cirtorch
    git pull
fi

# Setup package
pip install .

cd ../..