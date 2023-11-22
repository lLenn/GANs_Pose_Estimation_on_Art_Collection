import os
import json
import itertools
import cv2
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from utils import isArrayLike
from SWAHR.dataset.target_generators import HeatmapGenerator, JointsGenerator
from .HumanArt import HumanArt

# Split the set into keypoint and normal
class HumanArtKeypoints(Dataset):
    def __init__(self, root, categories=[], phase="train", removeImagesWithoutAnnotations=True, outputSize=[128, 256], sigma=2, maxNumberPeople=10, numberJoints=17, tagPerJoint=True):
        self.dataset = HumanArt(root, categories, phase, removeImagesWithoutAnnotations)
        self.numberJoints = numberJoints
        self.numberScales = len(outputSize)
        self.heatmapGenerator = [HeatmapGenerator(output_size, numberJoints, sigma) for output_size in outputSize]
        self.jointsGenerator = [JointsGenerator(maxNumberPeople, numberJoints, output_size, tagPerJoint) for output_size in outputSize]
             
    def __getitem__(self, index):
        image, target = self.dataset[index]
        
        joints = self.get_joints(target)
        joints_list = [joints.copy() for _ in range(self.numberScales)]
        target_list = list()

        if self.transforms:
            img, mask_list, joints_list = self.transforms(
                img, mask_list, joints_list
            )

        for scale_id in range(self.numberScales):
            target_t = self.heatmapGenerator[scale_id](joints_list[scale_id])
            joints_t = self.jointsGenerator[scale_id](joints_list[scale_id])

            target_list.append(target_t.astype(np.float32))
            joints_list[scale_id] = joints_t.astype(np.int32)

        return image, target_list, joints_list

    def get_joints(self, annotations):
        joints = np.zeros((len(annotations), self.numberJoints, 3))
        for i, annotation in enumerate(annotations):
            joints[i, :self.num_joints, :3] = np.array(annotation['keypoints']).reshape([-1, 3])
        return joints
          
    def __len__(self):
        return len(self.dataset.ids)
    
    def print(self):
        super().print()
        print("HumanArt keypoints dataset")