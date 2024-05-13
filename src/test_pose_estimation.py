import json
import os
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from pose_estimation.datasets import COCOSubset
from pose_estimation.metrics import AveragePrecision
from pose_estimation.networks import SWAHR, SWAHRConfig, ViTPose, ViTPoseConfig, ArtPose
from style_transfer.datasets import HumanArtDataset
from torchvision.transforms import transforms

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
    
def measure(rank, world_size, num_workers, batch_size, data_root, dataset, model, results_dir, results_prefix, config_file, options):
    init_distributed(rank, world_size)
    
    network = None
    config = None
    average_precision = None
    device = torch.device(f"cuda:{rank}")
    print(device)
    if model == "SWAHR":
        config = SWAHRConfig.create(config_file, options)
        network = SWAHR(results_prefix, config)
        network.loadModel(config.TEST.MODEL_FILE)
    elif model == "ViTPose":
        config = ViTPoseConfig.create(config_file, convertListToDict(options))
        network = ViTPose(config)
        network.loadModel(config.model_file)
    else:
        raise Exception("Model pose estimation not recognized")
    
    if "HumanArt" in data_root:
        dataset = HumanArtDataset(data_root, dataset, "test", False)
        average_precision = AveragePrecision(dataset.humanArt, results_dir, results_prefix)
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
        dataset = COCOSubset(data_root, dataset, "jpg")
        average_precision = AveragePrecision(dataset.coco, results_dir, results_prefix)
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
    
    metrics = {}
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader.dataset))
        for i, (images, annotations) in enumerate(dataloader):
            image_id = int(dataset.coco.loadImgs(dataset.ids[i])[0]["id"])
            image = images[0].numpy()
            _, _, final_results, scores = network.infer(rank, world_size, image, [torch.tensor(annotation["bbox"]).numpy() for annotation in annotations])
            predictions = []
            for idx in range(len(final_results)):
                predictions.append({
                    "keypoints": final_results[idx][:,:3].reshape(-1,).astype(float).tolist(),
                    "image_id": image_id,
                    "score": float(scores[idx]),
                    "category_id": 1
                })
            average_precision.process_predictions(rank, world_size, predictions)
            if len(final_results) > 0 and i%100 == 0:
                ArtPose.visualizePoseEstimation(image, np.delete(final_results, -1, 2).reshape(len(final_results), -1).astype(float).tolist(), scores, os.path.join(results_dir, results_prefix), image_id)
            pbar.update()

        pbar.close()
        
        metrics["average_precision"] = average_precision.get_average_precision(rank)
    
    if rank == 0:
        with open(os.path.join(results_dir, f"{results_prefix}_metrics.json"), "w") as result_file:
            result_file.write(json.dumps(metrics))
            
    close_distributed(rank, world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to the data root')
    parser.add_argument('--dataset', type=str, required=True, help='Relative path to the dataset config file')
    parser.add_argument('--model', type=str, required=True, help='Name of the model used')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use')
    parser.add_argument('--results_dir', type=str, default="", help='Path to results dir')
    parser.add_argument('--results_prefix', type=str, required=True, help='Prefix for result file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--options', help='Modify config options using the command-line', default=None, nargs="+")
    parser_args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    args = [
        world_size,
        parser_args.num_workers,
        parser_args.batch_size,
        parser_args.data_root,
        parser_args.dataset,
        parser_args.model,
        parser_args.results_dir,
        parser_args.results_prefix,
        parser_args.config,
        parser_args.options
    ]

    image_path = os.path.join(parser_args.results_dir, parser_args.results_prefix)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
       
    if world_size > 1:
        mp.spawn(measure, args, nprocs=world_size)
    else:
        measure(0, *args)