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
from available_algorithms import createDatasetIterator, createPoseEstimatorIterator, createStyleTransformerIterator
from pose_estimation.datasets import COCOSubset
from pose_estimation.networks import SWAHR, SWAHRConfig, ViTPose, ViTPoseConfig
from style_transfer.networks import StarGAN, StarGANConfig, AdaIN, AdaINConfig, CycleGAN, CycleGANConfig
from style_transfer.datasets import HumanArtDataset

def printNameValue(logger, name_value, fullArchName):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info("| Arch " + " ".join(["| {}".format(name) for name in names]) + " |")
    logger.info("|---" * (num_values + 1) + "|")

    if len(fullArchName) > 15:
        fullArchName = fullArchName[:8] + "..."
    logger.info(
        "| "
        + fullArchName
        + " "
        + " ".join(["| {:.3f}".format(value) for value in values])
        + " |"
    )

def worker(gpuIds, dataset, indices, poseEstimator, styleTransformer, logger, finalOutputDir, predictionQueue):
    artPose = ArtPose(poseEstimator, styleTransformer, True)
    artPose.loadModel()
            
    sub_dataset = torch.utils.data.Subset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(
        sub_dataset, sampler=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )
    
    predictions = artPose.validate(gpuIds, data_loader)    
    predictionQueue.put_nowait(predictions)
    
def benchmark(dataset, poseEstimator, styleTransformer, logger):
    datasetSize = len(dataset)
    predictionQueue = Queue(100)
    workers = []
    if NO_WORKERS > 1:
        for i in range(NO_WORKERS):
            index_groups = list(range(i, datasetSize, NO_WORKERS))
            process = Process(
                target = worker,
                args = (
                    0, dataset, index_groups, copy.deepcopy(poseEstimator), copy.deepcopy(styleTransformer), logger, "./.output", predictionQueue
                )
            )
            process.start()
            workers.append(process)
            logger.info("==>" + " Worker {} Started, responsible for {} images".format(i, len(index_groups)))
        
        allPredictions = []
        for _ in range(NO_WORKERS):
            allPredictions += predictionQueue.get()
        
        for process in workers:
            process.join()
    else:
        worker(0, dataset, range(datasetSize), copy.deepcopy(poseEstimator), copy.deepcopy(styleTransformer), logger, "./.output", predictionQueue)
        allPredictions = predictionQueue.get()
        
    resultFolder = ".output/results"
    if not os.path.exists(resultFolder):
        os.makedirs(resultFolder)
    resultFile = os.path.join(resultFolder, f"keypoints_SWAHR_results.json")

    json.dump(allPredictions, open(resultFile, 'w'))

    info_str = dataset.keypointEvaluation(resultFile)
    name_values = OrderedDict(info_str)
    
    if isinstance(name_values, list):
        for name_value in name_values:
            printNameValue(logger, name_value, "SWAHR")
    else:
        printNameValue(logger, name_values, "SWAHR")

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
    device = torch.device(f"cuda:{rank}")
    print(device)
    if model_pose_estimation == "SWAHR":
        pose_estimation_config = SWAHRConfig.create(config_file_pose_estimation, options_pose_estimation)
        pose_estimation = SWAHR(results_prefix, pose_estimation_config)
        pose_estimation.loadModel(pose_estimation_config.TEST.MODEL_FILE)
    elif model_pose_estimation == "ViTPose":
        pose_estimation_config = ViTPoseConfig.create(config_file_pose_estimation, options_pose_estimation)
        pose_estimation = ViTPose(pose_estimation_config)
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
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,    
            drop_last=False,
            sampler=None if world_size == 1 else DistributedSampler(dataset)
        )
    elif action == ArtPose.PHOTOGRAPHIC_TO_ARTISTIC:
        dataset = HumanArtDataset(data_root, dataset, "test", False)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,    
            drop_last=False,
            sampler=None if world_size == 1 else DistributedSampler(dataset),
        )
    else:
        raise Exception("Action not recognized")
    
    artPose = ArtPose(pose_estimation, style_transfer, True)
    predictions = artPose.validate(rank, world_size, dataloader, action) 
    
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
        
    if world_size > 1:
        mp.spawn(measure, args, nprocs=world_size)
    else:
        measure(0, *args)