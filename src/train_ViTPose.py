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
from mmpose.visualization import PoseLocalVisualizer
from mmengine.evaluator.evaluator import Evaluator
from mmpose.evaluation.metrics.coco_metric import CocoMetric
from mmengine.dataset.utils import pseudo_collate
from pose_estimation.networks import ViTPose, ViTPoseConfig

def main(parser_args):
    world_size = torch.cuda.device_count()
    
    if parser_args.method == "train":
        method = train
        args = (world_size, parser_args.name, parser_args.batch_size, parser_args.num_workers, parser_args.config_file, parser_args.annotation_file[0], parser_args.annotation_file[1])
    elif parser_args.method == "validate":
        method = validate
        args = (world_size, parser_args.batch_size, parser_args.num_workers, parser_args.data_root, parser_args.model, parser_args.log, parser_args.config_file, parser_args.annotation_file[0])
    elif parser_args.method == "infer":   
        method = infer
        args = (parser_args.model, parser_args.infer_file, parser_args.log, parser_args.results_dir, parser_args.config_file)
    
    if world_size > 1:
        mp.spawn(method, args, nprocs=world_size,)
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
    
    visualizer = PoseLocalVisualizer()
    prediction_image = visualizer._draw_instances_kpts(image, pred_instances, 0.3, False, 'mmpose')
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
    
    prefix = os.path.splitext(os.path.basename(annotation_file))[0]
    evaluator = Evaluator(CocoMetric(os.path.join("../../Datasets/", annotation_file), prefix=False))
    evaluator.dataset_meta = dataset.metainfo
    metrics = model.validate(rank, world_size, dataloader, evaluator)
    
    if rank == 0:
        json.dump(metrics, open(os.path.join(log, f"metrics_vitpose_{prefix}_{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')}.json"), 'w'))
    
    close_distributed(rank, world_size)
    
def train(rank, world_size, name, batch_size, num_workers, config_file, annotation_file_train, annotation_file_val):
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    config = ViTPoseConfig.create(config_file)
    config.experiment_name = "vitpose_" + name
    config.work_dir = f"../../Models/vitpose/{name}"
    config.resume = True
    config.load_from = None
    # config.launcher = "pytorch"
    config.auto_scale_lr.enable = True
    config.train_dataloader.batch_size = batch_size
    config.train_dataloader.num_workers = num_workers
    config.train_dataloader.dataset.ann_file = annotation_file_train
    config.val_dataloader.batch_size = batch_size//2 if batch_size > 1 else 1
    config.val_dataloader.num_workers = num_workers
    config.val_dataloader.dataset.ann_file = annotation_file_val
    config.val_evaluator.ann_file = os.path.join(config.data_root, annotation_file_val)
    config.default_hooks.checkpoint.out_dir = f"../../Models/vitpose"
    '''
    config.visualizer.vis_backends[1].init_kwargs.name = name + " vitpose"
    config.visualizer.vis_backends[1].init_kwargs.server = "http://116.203.134.130"
    config.visualizer.vis_backends[1].init_kwargs.env = "vitpose_" + name
    '''
    model = ViTPose(config)
    
    model.train()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="infer", help='The method to execute: infer, validate, train')
    parser.add_argument('--name', type=str, default="", help='The name of the training')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use')
    parser.add_argument('--num_workers', type=int, default=1, help='The number of workers for the dataloader')
    parser.add_argument('--log', type=str, default="", help='Path to log dir')
    parser.add_argument('--data_root', type=str, default="", help='The path to the data root')
    parser.add_argument('--results_dir', type=str, default="", help='Path to results dir')
    parser.add_argument('--model', type=str, default="", help='Path to pretrained model')
    parser.add_argument('--infer_file', type=str, default="", help='File to infer')
    parser.add_argument('--config_file', type=str, default="", help='Path to config file')
    parser.add_argument('--annotation_file', type=str, nargs='+', default="", help='File of the annotations relative from data root')
    parser_args = parser.parse_args()
    
    main(parser_args)