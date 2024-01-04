import argparse
import torch
import cv2
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import transforms
from pose_estimation.datasets import ArtPoseDataset
from pose_estimation.networks import ViTPose, ViTPoseConfig

def infer(gpu, model_path, image_path, log, config_file):
    config = ViTPoseConfig.create(config_file)
    config.model.pretrained = None
    config.data.test.test_mode = True
    config.work_dir = log
    model = ViTPose(config)
    model.loadModel(model_path)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((192, 256)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    image = transform(image)
    model.infer(image)

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
    parser.add_argument('--model', type=str, default="", help='Path to pretrained model')
    parser.add_argument('--infer_file', type=str, default="", help='File to infer')
    parser.add_argument('--config_file', type=str, default="", help='Path to log dir')
    parser.add_argument('--annotation_file', type=str, default="", help='File of the annotations')
    parser_args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    
    '''
    method = validate
    args = (world_size, parser_args.batch_size, parser_args.num_workers, parser_args.log, parser_args.config_file, parser_args.annotation_file)
    '''
    
    method = infer
    args = (parser_args.model, parser_args.infer_file, parser_args.log, parser_args.config_file)
    
    if world_size > 1:
        mp.spawn(method, args, nprocs=world_size)
    else:
        method(0, *args)