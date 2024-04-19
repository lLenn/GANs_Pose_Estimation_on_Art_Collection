import json
import os
import copy
import torch
import torch.multiprocessing as mp
import argparse
from collections import OrderedDict
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from multiprocessing import Process, Queue
from pose_estimation.networks.ArtPose import ArtPose
from pose_estimation.metrics import AveragePrecision
from pose_estimation.datasets import COCOSubset
from pose_estimation.networks import SWAHR, SWAHRConfig, ViTPose, ViTPoseConfig
from style_transfer.networks import StarGAN, StarGANConfig, AdaIN, AdaINConfig, CycleGAN, CycleGANConfig
from style_transfer.datasets import HumanArtDataset

torch.multiprocessing.set_sharing_strategy('file_system')

def convertListToDict(list):
   dict = {}
   for i in range(0, len(list), 2):
       dict[list[i]] = list[i + 1]
   return dict

def init_distributed(rank, world_size):
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank) 

def close_distributed(rank, world_size):
    if world_size > 1:
        destroy_process_group()
    
def measure(rank, world_size, num_workers, batch_size, action, data_root, dataset, model_pose_estimation, model_style_transfer, results_dir, results_prefix, config_file_pose_estimation, options_pose_estimation, config_file_style_transfer, options_style_transfer):
    init_distributed(rank, world_size)
    
    pose_estimation = None
    style_transfer = None
    pose_estimation_config = None
    style_transfer_config = None
    average_precision = None
    device = torch.device(f"cuda:{rank}")
    print(device)
    if model_pose_estimation == "SWAHR":
        pose_estimation_config = SWAHRConfig.create(config_file_pose_estimation, options_pose_estimation)
        pose_estimation = SWAHR(results_prefix, pose_estimation_config)
        pose_estimation.loadModel(pose_estimation_config.TEST.MODEL_FILE)
    elif model_pose_estimation == "ViTPose":
        pose_estimation_config = ViTPoseConfig.create(config_file_pose_estimation, convertListToDict(options_pose_estimation))
        pose_estimation = ViTPose(pose_estimation_config)
        pose_estimation.loadModel(pose_estimation_config.model_file)
    else:
        raise Exception("Model pose estimation not recognized")
    
    if model_style_transfer == "CycleGAN":
        # "style_transfer/config/cyclegan_test.yaml"
        style_transfer_config = CycleGANConfig.create(config_file_style_transfer, options=options_style_transfer, phase="test")
        style_transfer_config.defrost()
        style_transfer_config.gpu_ids = [rank]
        style_transfer_config.freeze()
        style_transfer = CycleGAN(style_transfer_config)
        style_transfer.loadModel({
            "G_A": style_transfer_config.G_A,
            "G_B": style_transfer_config.G_B
        })
    elif model_style_transfer == "AdaIN":
        # "style_transfer/config/adain.yaml"
        style_transfer_config = AdaINConfig.create(config_file_style_transfer, options=options_style_transfer)
        style_transfer_config.defrost()
        style_transfer_config.device = f"cuda:{rank}"
        style_transfer_config.freeze()
        style_transfer = AdaIN(style_transfer_config)
        style_transfer.loadModel({
            "vgg": style_transfer_config.vgg,
            "decoder": style_transfer_config.decoder
        })
    elif model_style_transfer == "StarGAN":
        # "style_transfer/config/stargan.yaml"
        style_transfer_config = StarGANConfig.create(config_file_style_transfer, options=options_style_transfer)
        style_transfer = StarGAN(style_transfer_config)
        style_transfer.to(device)
        style_transfer.loadModel(style_transfer_config.checkpoint_dir, "latest")
    else:
        raise Exception("Model not recognized") 
 
    if action == ArtPose.PHOTOGRAPHIC_TO_ARTISTIC:
        dataset = COCOSubset(data_root, dataset, "jpg")
        average_precision = AveragePrecision(dataset.coco, results_dir, results_prefix)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=True,    
            drop_last=False,
            sampler=None if world_size == 1 else DistributedSampler(dataset)
        )
    elif action == ArtPose.ARTISTIC_TO_PHOTOGRAPHIC:
        dataset = HumanArtDataset(data_root, dataset, "test", False)
        average_precision = AveragePrecision(dataset.humanArt, results_dir, results_prefix)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=True,    
            drop_last=False,
            sampler=None if world_size == 1 else DistributedSampler(dataset),
        )
    else:
        raise Exception("Action not recognized")
    
    metrics = {}
    def hook(predictions):
        average_precision.process_predictions(rank, world_size, predictions)
        
    artPose = ArtPose(pose_estimation, style_transfer, True)
    artPose.validate(rank, world_size, dataloader, action, os.path.join(results_dir, results_prefix), hook, 1024 if action == ArtPose.ARTISTIC_TO_PHOTOGRAPHIC else False)
    
    metrics["average_precision"] = average_precision.get_average_precision(rank)
    
    if rank == 0:
        with open(os.path.join(results_dir, f"{results_prefix}_metrics.json"), "w") as result_file:
            result_file.write(json.dumps(metrics))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, choices=[ArtPose.PHOTOGRAPHIC_TO_ARTISTIC, ArtPose.ARTISTIC_TO_PHOTOGRAPHIC], required=True, help=f'{ArtPose.PHOTOGRAPHIC_TO_ARTISTIC} or {ArtPose.ARTISTIC_TO_PHOTOGRAPHIC}')
    parser.add_argument('--data_root', type=str, required=True, help='Path to the data root')
    parser.add_argument('--dataset', type=str, required=True, help='Relative path to the dataset config file')
    parser.add_argument('--model_pose_estimation', type=str, required=True, help='Name of the model used')
    parser.add_argument('--model_style_transfer', type=str, required=True, help='Name of the model used')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use')
    parser.add_argument('--results_dir', type=str, default="", help='Path to results dir')
    parser.add_argument('--results_prefix', type=str, required=True, help='Prefix for result file')
    parser.add_argument('--config_pose_estimation', type=str, required=True, help='Path to pose estimation config file')
    parser.add_argument('--options_pose_estimation', help='Modify pose estimation config options using the command-line', default=None, nargs='+')
    parser.add_argument('--config_style_transfer', type=str, required=True, help='Path to style transfer config file')
    parser.add_argument('--options_style_transfer', help='Modify style transfer config options using the command-line', default=None, nargs='+')
    parserargs = parser.parse_args()
            
    world_size = torch.cuda.device_count()
    args = [
        world_size,
        parserargs.num_workers,
        parserargs.batch_size,
        parserargs.action,
        parserargs.data_root,
        parserargs.dataset,
        parserargs.model_pose_estimation,
        parserargs.model_style_transfer,
        parserargs.results_dir,
        parserargs.results_prefix,
        parserargs.config_pose_estimation,
        parserargs.options_pose_estimation,
        parserargs.config_style_transfer,
        parserargs.options_style_transfer
    ]
        
    image_path = os.path.join(parserargs.results_dir, parserargs.results_prefix)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
        
    if world_size > 1:
        mp.spawn(measure, args, nprocs=world_size)
    else:
        measure(0, *args)