import torch
import numpy as np
import mmpose.datasets.transforms
import pose_estimation.vitpose
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
        
    def _infer(self, rank, world_size, image, bbox):
        self.model.to(torch.device(f"cuda:{rank}"))
        data_list = []
        pipeline = Compose(self.config.val_pipeline)
        data_info = dict(img=image, bbox=np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]]))
        data_info['bbox_score'] = np.ones(1, dtype=np.float32)
        data_info.update(self.dataset_meta)
        data_list.append(pipeline(data_info))
        data_samples = self.model.test_step(pseudo_collate(data_list))
        return data_samples[0].pred_instances
    
    def infer(self, rank, world_size, image, bboxes):
        keypoints = []
        scores = []
        for bbox in bboxes:
            keypoint = []
            score = 0
            # vitpose gives different results if run multiple times.
            # the poses are inferred 10 times and the best is chosen.
            for _ in range(10):
                prediction = self._infer(rank, world_size, image, bbox)
                prediction_score = np.mean(prediction.keypoint_scores)
                if score < prediction_score:
                    keypoint = np.concatenate((prediction.keypoints[0], prediction.keypoint_scores.T, [[0]]*prediction.keypoints[0].shape[0]), axis=1)
                    score = prediction_score
            keypoints.append(keypoint)
            scores.append(score)
        return None, None, keypoints, scores
    
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
    
    def train(self):
        runner = Runner.from_cfg(self.config)
        runner.train()