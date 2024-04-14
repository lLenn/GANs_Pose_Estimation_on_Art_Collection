import logging
import argparse
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from pose_estimation.datasets import ArtPoseKeypoints
from pose_estimation.networks import SWAHR, SWAHRConfig

def main(rank, world_size, name, batch_size, num_workers, config_file, annotation_file, log_dir, logger):
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=600))
        torch.cuda.set_device(rank)
    
    config = SWAHRConfig.create(config_file, [])
    config.defrost()
    config.RANK = rank
    config.WORLD_SIZE = world_size
    config.PRINT_FREQ = 1000
    config.SAVE_FREQ = 1
    config.LOG_DIR = log_dir
    config.TRAIN.SAVE_NO = 2
    config.TRAIN.RESUME = True
    config.TRAIN.END_EPOCH = 200
    config.TRAIN.CHECKPOINT = "../../Models/swahr/saves"
    config.DATASET.ROOT = "../../Datasets/coco"
    config.VISDOM.NAME = name + " swahr"
    config.VISDOM.SERVER = "http://116.203.134.130"
    config.VISDOM.ENV = "swahr_" + name
    config.freeze()
    SWAHRConfig.configureEnvironment(config)
    network = SWAHR(name, config)

    dataset = ArtPoseKeypoints(config, annotation_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if world_size == 1 else False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        sampler=None if world_size == 1 else DistributedSampler(dataset)
    )

    network.train(rank, world_size, dataloader, logger)
    
    if world_size > 1:
        destroy_process_group()


if __name__ == '__main__':        
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="", help='The name of the training')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use')
    parser.add_argument('--num_workers', type=int, default=1, help='The number of workers for the dataloader')
    parser.add_argument('--log', type=str, default="", help='Path to log dir')
    parser.add_argument('--config_file', type=str, default="", help='Path to log dir')
    parser.add_argument('--annotation_file', type=str, default="", help='File of the annotations')
    args = parser.parse_args()
     
    if args.log != "":
        logging.basicConfig(
            filename=os.path.join(args.log, f"train_SWAHR_{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')}.log"),
            format='%(asctime)-15s %(message)s'
        )
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)
    else:
        logger = None
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            main,
            (world_size, args.name, args.batch_size, args.num_workers, args.config_file, args.annotation_file, args.log, logger),
            nprocs=world_size
        )
    else:
        main(0, world_size, args.name, args.batch_size, args.num_workers, args.config_file, args.annotation_file, args.log, logger)