import logging
import argparse
import os
from datetime import datetime
from torch.utils.data import DataLoader
from pose_estimation.datasets import ArtPoseKeypoints
from pose_estimation.networks import SWAHR, SWAHRConfig

def main(name, gpu, batch_size, num_workers, config_file, annotation_file, log_dir, logger):
    config = SWAHRConfig.create(config_file, [])
    config.defrost()
    config.WORLD_SIZE = 1
    config.PRINT_FREQ = 100
    config.SAVE_FREQ = 1
    config.LOG_DIR = log_dir
    config.TRAIN.SAVE_NO = 2
    config.TRAIN.RESUME = True
    config.TRAIN.END_EPOCH = 300
    config.TRAIN.CHECKPOINT = "../../Models/swahr/checkpoints"
    config.DATASET.ROOT = "../../Datasets/coco"
    config.VISDOM.NAME = name + " swahr"
    config.VISDOM.SERVER = "http://116.203.134.130"
    config.VISDOM.ENV = "swahr_" + name
    config.freeze()
    SWAHRConfig.configureEnvironment(config)
    network = SWAHR(name, config)

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    dataset = ArtPoseKeypoints(config, annotation_file)
    dataloader =  DataLoader(   
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )

    network.train(dataloader, logger)


if __name__ == '__main__':        
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="", help='The name of the training')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use')
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
        
    main(args.name, args.gpu_id, args.batch_size, args.num_workers, args.config_file, args.annotation_file, args.log, logger)