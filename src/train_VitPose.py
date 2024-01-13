import os
import argparse
import torch
import cv2
import torch.multiprocessing as mp
import numpy as np
import json
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from mmpose.datasets.datasets.body import HumanArtDataset
from mmengine.evaluator.evaluator import Evaluator
from mmpose.evaluation.metrics.coco_metric import CocoMetric
from mmengine.dataset.utils import pseudo_collate
from pose_estimation.networks import ViTPose, ViTPoseConfig, ViTPoseVisualizer
from pose_estimation.datasets import MMArtPoseDataset

def main(parser_args):
    world_size = torch.cuda.device_count()
    
    method = validate
    args = (world_size, parser_args.batch_size, parser_args.num_workers, parser_args.data_root, parser_args.model, parser_args.log, parser_args.config_file, parser_args.annotation_file)
   
    '''
    method = infer
    args = (world_size, parser_args.model, parser_args.infer_file, parser_args.log, parser_args.results_dir, parser_args.config_file)
    post_process = None
    '''
    
    if world_size > 1:
        mp.spawn(method, args, nprocs=world_size, join=False)
    else:
        method(0, *args)

def init_distributed(rank, world_size):
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank) 

def close_distributed(rank, world_size):
    if world_size > 1:
        destroy_process_group()

def infer(gpu, model_path, image_path, log, results_dir, config_file):    
    config = ViTPoseConfig.create(config_file)
    config.model.backbone.init_cfg = None
    model = ViTPose(config)
    model.loadModel(model_path)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w = image.shape[:2]
    bbox = np.array([0, 0, w, h], dtype=np.float32)
    
    pred_instances = model.infer(gpu, image, bbox)
    
    visualizer = ViTPoseVisualizer(log, config.visdom)
    prediction_image = visualizer.draw_predictions(image, pred_instances)
    prediction_image = cv2.cvtColor(prediction_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(results_dir, "vit_inference.png"), prediction_image)

def validate(rank, world_size, batch_size, num_workers, data_root, model_path, log, config_file, annotation_file):
    init_distributed(rank, world_size)
    
    config = ViTPoseConfig.create(config_file)
    config.model.backbone.init_cfg = None
    model = ViTPose(config)
    model.loadModel(model_path)
    
    dataset = HumanArtDataset(annotation_file, data_root=data_root, pipeline=config.val_pipeline, test_mode=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,    
        drop_last=False,
        sampler=None if world_size == 1 else DistributedSampler(dataset),
        collate_fn=pseudo_collate
    )
    
    # visualizer = ViTPoseVisualizer(log, config.visdom)
    # visualizer.dataset_meta = datset.metainfo
    prefix = os.path.splitext(os.path.basename(annotation_file))[0]
    evaluator = Evaluator(CocoMetric(os.path.join("../../Datasets/", annotation_file), prefix=False))
    evaluator.dataset_meta = dataset.metainfo
    metrics = model.validate(rank, world_size, dataloader, evaluator)
    
    if rank == 0:
        json.dump(metrics, open(os.path.join(log, f"metrics_vitpose_{prefix}_{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')}.json"), 'w'))
    
    close_distributed(rank, world_size)
    
def train(rank, world_size, batch_size, num_workers, data_root, log, config_file, annotation_file):
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(rank)

    
    config = ViTPoseConfig.create(config_file)
    config.visdom.name = "vitpose"
    config.visdom.env = "test_vitpose"
    config.work_dir = "../../Models/vitpose/checkpoints"
    config.save_no = 2
    config.resume = True
    config.load_from = None    
    config.auto_scale_lr.enable = True
    model = ViTPose(config)
    
    dataset = MMArtPoseDataset(annotation_file, data_root=data_root, pipeline=config.train_pipeline, data_prefix=dict(img='train/'))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,    
        drop_last=False,
        sampler=None if world_size == 1 else DistributedSampler(dataset),
        collate_fn=pseudo_collate
    )
    
    visualizer = ViTPoseVisualizer(log, config.visdom)
    visualizer.dataset_meta = dataset.metainfo
    
    model.train(rank, world_size, dataloader, visualizer)
    
    close_distributed(rank, world_size)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="", help='The name of the training')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use')
    parser.add_argument('--num_workers', type=int, default=1, help='The number of workers for the dataloader')
    parser.add_argument('--log', type=str, default="", help='Path to log dir')
    parser.add_argument('--data_root', type=str, default="", help='The path to the data root')
    parser.add_argument('--results_dir', type=str, default="", help='Path to results dir')
    parser.add_argument('--model', type=str, default="", help='Path to pretrained model')
    parser.add_argument('--infer_file', type=str, default="", help='File to infer')
    parser.add_argument('--config_file', type=str, default="", help='Path to config file')
    parser.add_argument('--annotation_file', type=str, default="", help='File of the annotations relative from data root')
    parser_args = parser.parse_args()
    
    main(parser_args)