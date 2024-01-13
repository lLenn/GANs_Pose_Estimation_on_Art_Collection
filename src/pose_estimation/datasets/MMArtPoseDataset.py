import os
from mmpose.datasets.datasets import CocoDataset

class MMArtPoseDataset(CocoDataset):
        
    def _get_anno_file_name(self):
        return os.path.join(self.root, "annotations", self.file)
    
    METAINFO: dict = dict(from_file='src/pose_estimation/config/coco.py')