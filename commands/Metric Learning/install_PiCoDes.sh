#!/bin/bash

bash ./commands/prep_environment.sh

cd lib

# Clone the PiCoDes API
if [ ! -d PiCoDes ]
then
    git clone https://github.com/megvii-research/PiCoDes.git PiCoDes
    cd PiCoDes
else
    cd PiCoDes
    git pull
fi

# Build
make

cd ../..