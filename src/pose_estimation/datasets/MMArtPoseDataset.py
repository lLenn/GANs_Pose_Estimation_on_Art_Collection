from mmpose.datasets.datasets import CocoDataset

class MMArtPoseDataset(CocoDataset):    
    METAINFO: dict = dict(from_file='src/pose_estimation/config/coco.py')