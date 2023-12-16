#!/bin/bash

bash ./commands/prep_environment.sh

cd lib

# Clone the StarGAN API
if [ ! -d StarGAN ]
then
    git clone https://github.com/lLenn/stargan-v2.git StarGAN
    cd StarGAN
else
    cd StarGAN
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
cd StarGAN

cd ..