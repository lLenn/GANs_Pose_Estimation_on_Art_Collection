#!/bin/bash

bash ./commands/prep_environment.sh

cd lib

# Clone the UGATIT API
if [ ! -d UGATITLib ]
then
    git clone https://github.com/lLenn/UGATIT-pytorch.git UGATITLib
    cd UGATITLib
else
    cd UGATITLib
fi

if [ ! -f "__init__.py" ]
then
    touch "__init__.py"
fi

# Setup package
cd ..
cp ../commands/Style\ Transfer/setup.py ./setup.py
python3 ./setup.py develop

cd ..