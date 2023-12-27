import os, cv2
import numpy as np
from SWAHR.dataset.COCOKeypoints import CocoKeypoints
from SWAHR.dataset.transforms import build_transforms
from SWAHR.dataset.target_generators import HeatmapGenerator
from SWAHR.dataset.target_generators import ScaleAwareHeatmapGenerator
from SWAHR.dataset.target_generators import JointsGenerator

ID_ADDITION = 7000000000000
ID_SUB_ADDITION = 100000000000

class ArtPoseKeypoints(CocoKeypoints):
    def __init__(self, cfg, file):
        transforms = build_transforms(cfg, True)

        if cfg.DATASET.SCALE_AWARE_SIGMA:
            _HeatmapGenerator = ScaleAwareHeatmapGenerator
        else:
            _HeatmapGenerator = HeatmapGenerator

        heatmap_generator = [_HeatmapGenerator(output_size, cfg.DATASET.NUM_JOINTS, cfg.DATASET.SIGMA) for output_size in cfg.DATASET.OUTPUT_SIZE]
        joints_generator = [JointsGenerator(cfg.DATASET.MAX_NUM_PEOPLE, cfg.DATASET.NUM_JOINTS, output_size, cfg.MODEL.TAG_PER_JOINT) for output_size in cfg.DATASET.OUTPUT_SIZE]
        
        self.file = file
        
        super().__init__(cfg, cfg.DATASET.TRAIN, True, heatmap_generator, joints_generator, transforms)
        
    def _get_anno_file_name(self):
        return os.path.join(self.root, "annotations", self.file)
    
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        
        image_info = coco.loadImgs(img_id)[0]
        file_name = image_info['file_name']

        if self.data_format == 'zip':
            img = zipreader.imread(
                self._get_image_path(file_name, image_info["id"]>=ID_SUB_ADDITION),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            img = cv2.imread(
                self._get_image_path(file_name, image_info["id"]>=ID_SUB_ADDITION),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        mask = self.get_mask(target, index)

        anno = [
            obj for obj in target
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]

        # TODO(bowen): to generate scale-aware sigma, modify `get_joints` to associate a sigma to each joint
        joints = self.get_joints(anno)

        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]
        target_list = list()

        if self.transforms:
            img, mask_list, joints_list = self.transforms(
                img, mask_list, joints_list
            )

        for scale_id in range(self.num_scales):
            target_t = self.heatmap_generator[scale_id](joints_list[scale_id])
            joints_t = self.joints_generator[scale_id](joints_list[scale_id])

            target_list.append(target_t.astype(np.float32))
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)
            joints_list[scale_id] = joints_t.astype(np.int32)

        return img, target_list, mask_list, joints_list
    
    
    def _get_image_path(self, file_name, styled):
        images_dir = os.path.join(self.root, 'images')
        dataset = 'test2017' if 'test' in self.dataset else self.dataset
        if styled:
            dataset = dataset[:-4]
            
        if self.data_format == 'zip':
            return os.path.join(images_dir, dataset) + '.zip@' + file_name
        else:
            return os.path.join(images_dir, dataset, file_name)