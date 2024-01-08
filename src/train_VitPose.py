import os
import argparse
import torch
import cv2
import torch.multiprocessing as mp
import mmcv
import mmpose.datasets.transforms
import numpy as np
from mmengine.structures import InstanceData
from mmengine.dataset import Compose, pseudo_collate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import transforms
from pose_estimation.datasets import ArtPoseDataset
from pose_estimation.networks import ViTPose, ViTPoseConfig, ViTPoseVisualizer
from mmpose.datasets.datasets.utils import parse_pose_metainfo
import mmpose.datasets.datasets
from mmpose.registry import DATASETS

def infer(gpu, model_path, image_path, log, results_dir, config_file):
    config = ViTPoseConfig.create(config_file)
    config.model.backbone.init_cfg = None
    model = ViTPose(config)
    model.loadModel(model_path)
    
    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((192, 256)),
        transforms.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
    ])
    
    transformed_image = transform(image).cuda(gpu).unsqueeze(0)
    keypoints, scores = model.infer(gpu, transformed_image)
    instance = InstanceData(keypoints=keypoints, keypoint_scores=scores)
    '''
    dataset_meta = parse_pose_metainfo(DATASETS.get(config.val_dataloader.dataset.type).METAINFO)
    
    image = mmcv.imread(image_path, channel_order='rgb')
    h, w = image.shape[:2]
    bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
    
    data_list = []
    pipeline = Compose(config.val_pipeline)
    data_info = dict(img_path=image_path, bbox=bboxes)
    data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
    data_info.update(dataset_meta)
    data_list.append(pipeline(data_info))
    data_samples = model.model.test_step(pseudo_collate(data_list))
    
    visualizer = ViTPoseVisualizer(log, config.visdom)
    prediction_image = visualizer.draw_predictions(image, data_samples[0].pred_instances)
    prediction_image = cv2.cvtColor(prediction_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(results_dir, "vit_inference.png"), prediction_image)

def validate(gpu, world_size, batch_size, num_workers, log, config_file, annotation_file):
    config = ViTPoseConfig.create(config_file)
    config.model.pretrained = None
    config.data.test.test_mode = True
    config.work_dir = log
    model = ViTPose(config)
    
    dataset = ArtPoseDataset("../../Datasets/coco", "coco", "person_keypoints_val2017")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if world_size == 1 else False,
        num_workers=num_workers,
        pin_memory=config.data.pin_memory,
        sampler=None if world_size == 1 else DistributedSampler(dataset)
    )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use')
    parser.add_argument('--num_workers', type=int, default=1, help='The number of workers for the dataloader')
    parser.add_argument('--log', type=str, default="", help='Path to log dir')
    parser.add_argument('--results_dir', type=str, default="", help='Path to results dir')
    parser.add_argument('--model', type=str, default="", help='Path to pretrained model')
    parser.add_argument('--infer_file', type=str, default="", help='File to infer')
    parser.add_argument('--config_file', type=str, default="", help='Path to config file')
    parser.add_argument('--annotation_file', type=str, default="", help='File of the annotations')
    parser_args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    
    '''
    method = validate
    args = (world_size, parser_args.batch_size, parser_args.num_workers, parser_args.log, parser_args.config_file, parser_args.annotation_file)
    '''
    
    method = infer
    args = (parser_args.model, parser_args.infer_file, parser_args.log, parser_args.results_dir, parser_args.config_file)
    
    if world_size > 1:
        mp.spawn(method, args, nprocs=world_size)
    else:
        method(0, *args)