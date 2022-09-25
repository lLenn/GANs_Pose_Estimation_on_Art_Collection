#!/bin/bash

pip install gdown

cd bin
if [[ ! -f "AlphaPose/detector/yolo/data/yolov3-spp.weights" ]]; then
    mkdir AlphaPose/detector/yolo/data -p
    gdown 1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC -O AlphaPose/detector/yolo/data/yolov3-spp.weights
fi
if [[ ! -f "AlphaPose/detector/tracker/data/JDE-1088x608-uncertainty" ]]; then
    mkdir AlphaPose/detector/tracker/data -p
    gdown 1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA -O AlphaPose/detector/tracker/data/JDE-1088x608-uncertainty
fi
if [[ ! -f "AlphaPose/pretrained_models/fast_res50_256x192.pth" ]]; then
    gdown 1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn -O AlphaPose/pretrained_models/fast_res50_256x192.pth
fi
if [[ ! -f "AlphaPose/detector/yolox/data/yolox_x.pth" ]]; then
    wget -P AlphaPose/detector/yolox/data/ https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_x.pth
fi

cd AlphaPose
python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/ --save_img