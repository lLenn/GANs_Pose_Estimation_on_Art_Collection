#!/bin/bash

bash ./commands/prep_environment.sh

cd lib

# Clone the DPGN API
if [ ! -d DPGN ]
then
    git clone https://github.com/megvii-research/DPGN.git DPGN
    cd DPGN
else
    cd DPGN
    git pull
fi

if [ ! -f "__init__.py" ]
then
    touch "__init__.py"
fi

# Setup package
cd ..
cp ../commands/Metric\ Learning/setup.py ./setup_metric_learning.py
python3 ./setup_metric_learning.py develop
cd DPGN

cd ../..