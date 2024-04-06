import json
import os
import argparse
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from style_transfer.datasets import ImageDirectoryDataset
from style_transfer.networks import StarGAN, StarGANConfig
from style_transfer.networks import AdaIN, AdaINConfig
from style_transfer.networks import CycleGAN, CycleGANConfig
from style_transfer.metric import PerceptualDistance, InceptionScore, FrechetInceptionDistance, LearnedPerceptualImagePatchSimilarity

def init_distributed(rank, world_size):
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank) 

def close_distributed(rank, world_size):
    if world_size > 1:
        destroy_process_group()
    
def measure(rank, world_size, num_workers, batch_size, dataset_directory, real_directory, model, results_dir, results_prefix, config_file, options):
    init_distributed(rank, world_size)
    
    network = None
    config = None
    if model == "CycleGAN":
        # "style_transfer/config/cyclegan_test.yaml"
        config = CycleGANConfig.create(config_file, options=options, phase="test")
        network = CycleGAN(config)
        network.loadModel({
            "G_A": config.G_A,
            "G_B": config.G_B
        })
    elif model == "AdaIN":
        # "style_transfer/config/adain.yaml"
        config = AdaINConfig.create(config_file)
        network = AdaIN(config)
        network.loadModel({
            "vgg": config.vgg,
            "decoder": config.decoder
        })
    elif model == "StarGAN":
        # "style_transfer/config/stargan.yaml"
        config = StarGANConfig.create(config_file)
        network = StarGAN(config)
        network.loadModel(config.checkpoint_dir, "latest")
    
    size = 512
    
    dataset = ImageDirectoryDataset(dataset_directory, size=size)
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
    real_dataset = ImageDirectoryDataset(real_directory)
    real_dataloader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,    
        drop_last=False,
        sampler=None if world_size == 1 else DistributedSampler(real_dataset)
    )
    
    metrics = {}
    
    with torch.no_grad():
        real_iter = iter(real_dataset)
        perceptual_distance = PerceptualDistance()
        inception_score = InceptionScore()
        frechet_inception_distance = FrechetInceptionDistance()
        lpips = LearnedPerceptualImagePatchSimilarity(size=size)
        
        pbar = tqdm(total=len(dataloader))
        for data_batch in dataloader:
            style_image = None
            try:
                style_image = next(real_iter)
            except:
                real_iter = iter(real_dataset)
                style_image = next(real_iter)
            style_image = style_image.cuda()
            
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).cuda()
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).cuda()
            
            if model == "CycleGAN":
                data_batch = data_batch.cuda()
                data_batch = (data_batch - mean) / std
                predictions = network.photographicToArtistic(data_batch)
                data_batch = (data_batch * std) + mean
                predictions = (predictions * std) + mean
            elif model == "AdaIN":
                data_batch = data_batch.cuda()
                predictions = network.transformTo(data_batch, style_image)
            elif model == "StarGAN":
                data_batch = data_batch.cuda()
                data_batch = (data_batch - mean) / std
                i = 0
                if config.style_output == "baroque":
                    i = 1
                elif config.style_output == "impressionism":
                    i = 2
                elif config.style_output == "renaissance":
                    i = 3
                style = torch.tensor([i]).cuda()
                predictions = network.imageToStyle(data_batch, style)
                data_batch = (data_batch * std) + mean
                predictions = (predictions * std) + mean
            
            perceptual_distance.process_images(data_batch, style_image, predictions)
            inception_score.process_generated_images(predictions)
            frechet_inception_distance.process_generated_images(predictions)
            lpips.process_generated_images(predictions)
            pbar.update()
        pbar.close()
              
        pbar = tqdm(total=len(real_dataloader))
        for real_batch in real_dataloader:
            real_batch = real_batch.cuda()
            frechet_inception_distance.process_real_images(real_batch)
            lpips.process_real_images(real_batch)
            pbar.update()
        pbar.close()
    
    metrics["perceptual_distance"] = perceptual_distance.get_perceptual_distance(rank, world_size)
    inception_score_mean, inceptions_score_stddev = inception_score.get_inception_score(rank, world_size)
    metrics["inception_score"] = {
        "mean": inception_score_mean,
        "standard_deviation": inceptions_score_stddev
    }
    metrics["frechet_inception_distance"] = frechet_inception_distance.get_frechet_inception_distance(rank, world_size)
    lpips_similarity, lpips_variation = lpips.get_lpips(rank, world_size)
    metrics["lpips"] = {
        "similarity": lpips_similarity,
        "variation": lpips_variation
    }
    
    with open(os.path.join(results_dir, f"{results_prefix}_metrics"), "w") as result_file:
        result_file.write(json.dumps(metrics))
    
    close_distributed(rank, world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset to transform to style')
    parser.add_argument('--real', type=str, required=True, help='Path to real images of style')
    parser.add_argument('--model', type=str, required=True, help='Name of the model used')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use')
    parser.add_argument('--results_dir', type=str, default="", help='Path to results dir')
    parser.add_argument('--results_prefix', type=str, required=True, help='Prefix for result file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('opts', help='Modify config options using the command-line', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
            
    world_size = torch.cuda.device_count()
        
    if world_size > 1:
        mp.spawn(measure, [world_size, args.num_workers, args.batch_size, args.dataset, args.real, args.model, args.results_dir, args.results_prefix, args.config, args.opts], nprocs=world_size)
    else:
        measure(0, world_size, args.num_workers, args.batch_size, args.dataset, args.real, args.model, args.results_dir, args.results_prefix, args.config, args.opts)