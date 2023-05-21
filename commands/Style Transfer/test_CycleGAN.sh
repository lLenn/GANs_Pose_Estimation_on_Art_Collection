#!/bin/bash

cd ./lib/CycleGAN

python3 ./test.py --dataroot ../../../data/cezanne2photo --name style_cezanne_pretrained --model test --no_dropout

cd ../..