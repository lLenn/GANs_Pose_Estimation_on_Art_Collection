import torch
import numpy as np
import mmpose.datasets.transforms
from tqdm import tqdm
from collections import deque
from mmengine.dataset import Compose, pseudo_collate
from mmpose.models import build_pose_estimator
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.registry import DATASETS
from mmengine.runner import Runner

class ViTPose:
    def __init__(self, config):
        self.config = config
        self.model = build_pose_estimator(config.model)
        self.dataset_meta = parse_pose_metainfo(DATASETS.get(config.val_dataloader.dataset.type).METAINFO)
        maxlen = (None if config.get("save_no", -1) < 0 else config.save_no)
        self.savedFiles = deque(maxlen = maxlen)
        
    def loadModel(self, file):
        json = torch.load(file)
        self.model.load_state_dict(json["state_dict"], strict=True)
        self.savedFiles.append(file)
        
    def infer(self, rank, image, bbox):
        data_list = []
        pipeline = Compose(self.config.val_pipeline)
        data_info = dict(img=image, bbox=bbox[None,])
        data_info['bbox_score'] = np.ones(1, dtype=np.float32)
        data_info.update(self.dataset_meta)
        data_list.append(pipeline(data_info))
        data_samples = self.model.test_step(pseudo_collate(data_list))
        return data_samples[0].pred_instances
    
    def validate(self, rank, world_size, data_loader, evaluator):
        pbar = tqdm(total=len(data_loader))
        self.model.eval()
        with torch.no_grad():
            for data_batch in data_loader:
                samples = self.model.val_step(data_batch)
                evaluator.process(data_samples=samples, data_batch=data_batch)
                pbar.update()
                    
            pbar.close()
            return evaluator.evaluate(len(data_loader.dataset))
    
    def train(self, rank, world_size, data_loader, visualizer):
        runner = Runner.from_cfg(self.config)
        runner.train()