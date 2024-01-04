import torch
from .vitpose import *
from collections import deque
from mmpose.models import build_pose_estimator

class ViTPose:
    def __init__(self, config):
        self.config = config
        self.model = build_pose_estimator(config.model)
        
        maxlen = (None if config.get("save_no", -1) < 0 else config.save_no)
        self.savedFiles = deque(maxlen = maxlen)
        
    def loadModel(self, file):
        json = torch.load(file)
        self.model.load_state_dict(json["state_dict"], strict=True)
        self.savedFiles.append(file)
        
    def infer(self, rank, image):
        self.model.cuda(rank)
        self.model.eval()
        with torch.no_grad():
            return self.model(image)