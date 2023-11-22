import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset
from utils import isArrayLike
from .HumanArt import HumanArt
from .HumanArtEvaluation import HumanArtEvaluation

# Split the set into keypoint and normal
class HumanArtDataset(Dataset):
    def __init__(self, root, category="", phase="train", removeImagesWithoutAnnotations=True):
        self.root = root
        self.category = category
        self.separator = "/"
        self.humanArt = HumanArt(self._get_annotation_file_name(category, phase))
        self.ids = list(self.humanArt.images.keys())               
        if removeImagesWithoutAnnotations:
            self.ids = [image_id for image_id in self.ids if len(self.humanArt.getAnnotationIds(imageIds=image_id, isCrowd=None)) > 0]
    
    def _get_annotation_file_name(self, category="", phase="training"):
        return os.path.join(
            self.root,
            "annotations",
            f"{phase}_humanart{'' if category == '' else '_' + category}.json"
        )

    def _get_image_path(self, relative):
        return os.path.normpath(os.path.join(self.root, *relative.split(self.separator)[1:]))
    
    def __getitem__(self, index):
        image_id = self.ids[index]
        annotation_ids = self.humanArt.getAnnotationIds(imageIds=image_id)
        target = self.humanArt.loadAnnotations(annotation_ids)
        
        image_info = self.humanArt.loadImages(image_id)[0]
        file_name = self._get_image_path(image_info['file_name'])
        
        image = cv2.imread(file_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        target = [annotation for annotation in target if annotation["iscrowd"] == 0 or annotation["num_keypoints"] > 0]
        
        return image, target
          
    def __len__(self):
        return len(self.ids)
           
    def keypointEvaluation(self, resultFile):       
        humanArt = self.humanArt.loadResults(resultFile)
        humanArtEvaluation = HumanArtEvaluation(self.humanArt, humanArt, 'keypoints')
        humanArtEvaluation.evaluate()
        humanArtEvaluation.accumulate()
        humanArtEvaluation.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, humanArtEvaluation.stats[ind]))

        return info_str
    
    def print(self):
        super().print()
        print("HumanArt dataset")