Set-Location lib\SWAHR-HumanPose

python .\tools\dist_valid.py --world_size 2 --cfg ../../study/models/w32_512_adam_lr1e-3.yaml TEST.MODEL_FILE ./models/pose_coco/pose_higher_hrnet_w32_512.pth DATASET.ROOT ../../../data/coco_data/

Set-Location ..\..