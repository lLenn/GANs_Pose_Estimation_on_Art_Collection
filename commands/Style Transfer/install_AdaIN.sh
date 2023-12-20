#!/bin/bash

bash ./commands/prep_environment.sh

cd lib

# Clone the AdaIN API
if [ ! -d AdaIN ]
then
    git clone https://github.com/lLenn/pytorch-AdaIN.git AdaIN
    cd AdaIN
else
    cd AdaIN
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
cd AdaIN

cd ../..