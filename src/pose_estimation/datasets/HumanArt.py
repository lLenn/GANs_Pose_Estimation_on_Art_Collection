# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from mmpose.datasets.datasets.base import BaseCocoStyleDataset


class HumanArtDataset(BaseCocoStyleDataset):
    METAINFO: dict = dict(from_file='src/pose_estimation/config/humanart.py')