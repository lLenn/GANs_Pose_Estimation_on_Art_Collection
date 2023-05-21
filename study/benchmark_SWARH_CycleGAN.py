from SWAHR.lib.config import cfg
from CycleGAN.models import create_model

# python .\tools\dist_valid.py --world_size 2 --cfg .\experiments\coco\higher_hrnet\w32_512_adam_lr1e-3.yaml TEST.MODEL_FILE .\models\pose_coco\pose_higher_hrnet_w32_512.pth


print(cfg)
print(create_model)
