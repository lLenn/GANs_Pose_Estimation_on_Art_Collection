#!/bin/bash

cd lib/SWAHR-HumanPose

python3 tools/dist_valid.py --world_size 1 --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml TEST.MODEL_FILE models/pose_coco/pose_higher_hrnet_w32_512.pth

cd ../..