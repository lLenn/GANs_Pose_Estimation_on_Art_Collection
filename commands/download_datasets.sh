#!/bin/bash

IFS="/" read -ra PWD_ARR <<< $(pwd)
if [[ "${PWD_ARR[-1]}" != "GANs_Pose_Estimation_on_Art_Collection" ]]; then
    echo "Execute commands from the root directory"
    exit
fi

cd ./lib/CycleGAN
bash ./datasets/download_cyclegan_dataset.sh cezanne2photo ../../../data