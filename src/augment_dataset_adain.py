import json
import os
import logging
import copy
import argparse
import cv2
import sys
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torchvision.transforms import transforms
from multiprocessing import Process, Queue
from SWAHR.dataset.COCODataset import CocoDataset
from style_transfer.networks import AdaIN, AdaINConfig
from utils.Augmentation import copyAnnotations, copyMetadata, saveImage, createDatasets

ID_ADDITION = 8000000000000
ID_SUB_ADDITION = 100000000000

def worker(gpuId, dataset, target, style_dir, styles, workerIndices, logger, styledQueue):    
    # Initiate the AdaIN model with the proper configuration
    config = AdaINConfig.create("style_transfer/config/adain.yaml")
    config.defrost()
    config.preserve_color = False
    config.alpha = 1.0
    config.device = f"cuda:{gpuId}"
    config.freeze()
    style_transfer = AdaIN(config)
    style_transfer.loadModel({
        "vgg": "../../Models/AdaIN/vgg_normalised.pth",
        "decoder": "../../Models/AdaIN/decoder.pth"
    })
    
    # Each style found at the given path will be iterated
    style_list = [file for file in os.listdir(style_dir) if os.path.isdir(os.path.join(style_dir, file)) and file in styles]
    indicesSize = len(workerIndices)
    
    # Array that will be passed back to the main process
    styled_images = []
    
    pbar = tqdm(total=indicesSize*len(style_list)) if logger is not None else None
    # Each model is iterated and the workerIndices are split equally so 
    for modelIndex, model in enumerate(style_list):
        if model not in styles:
            if logger is not None:
                pbar.update(indicesSize)
            continue
        id_addition = ID_ADDITION + ID_SUB_ADDITION * modelIndex
        
        # Load a list of images that can be used for the style transfer
        style_images = [file for file in os.listdir(os.path.join(style_dir, model)) if os.path.isfile(os.path.join(style_dir, model, file))]
        len_style_images = len(style_images)
        for imageIndex in workerIndices:
            # Copy the metadata and the annotations
            img_id = dataset.ids[imageIndex]
            metadata = dataset.coco.loadImgs(img_id)[0]
            file_path_from = os.path.join(dataset.root, 'images', dataset.dataset, metadata['file_name'])
            copiedMetadata = copyMetadata(metadata, id_addition)
            copiedAnnotations = copyAnnotations(dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=metadata["id"])), id_addition)
            
            # Check if the images wasn't already created in a previous execution
            file_path_to = os.path.join(dataset.root, 'images', target, copiedMetadata["file_name"])
            if not os.path.exists(file_path_to):
                # Load random style image
                style_path = os.path.join(style_dir, model, style_images[random.randint(0, len_style_images-1)])
                style = cv2.imread(style_path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
                style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
                image = cv2.imread(file_path_from, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Transform style to fit dimensions of content
                smallest_size = sys.maxsize
                for val in image.shape[:2]:
                    if val < smallest_size:
                        smallest_size = val
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(smallest_size),
                ])
                style = transform(style)
                image = transform(image)
                styledImage = style_transfer.transformTo(image, style).squeeze()
                saveImage(file_path_to, styledImage)
            styled_images.append({
                "style": model,
                "metadata": copiedMetadata,
                "annotations": copiedAnnotations
            })            
            if logger is not None:
                pbar.update()
            elif imageIndex % 1000 == 0:
                print(f"Worker {gpuId} finished {modelIndex+1} images of {model}")
        if logger is None:
            print(f"Worker {gpuId} finished {model}")
    
    if logger is not None:
        pbar.close()
    print(f"Worker {gpuId} closing ...")
    styledQueue.put_nowait(styled_images)
    print(f"Worker {gpuId} closed")
    
def augment(coco, source, target, style_dir, styles, gpu_ids, num_workers, logger):
    dataset = CocoDataset(coco, source, "jpg")
    datasetSize = len(dataset)
    workers = []
    
    # Multiprocess queue
    styledQueue = Queue(100)
    styleList = []
    
    # Each given gpu starts given num. workers to create style images
    num_gpus = len(gpu_ids)
    if num_gpus > 1 or num_workers > 1:
        for i in range(num_gpus*num_workers):
            index_groups = list(range(i, datasetSize, num_gpus*num_workers))
            process = Process(
                target = worker,
                args = (
                    gpu_ids[i//num_workers], dataset, target, style_dir, styles, index_groups, logger, styledQueue
                )
            )
            process.start()
            workers.append(process)
            print(f"==> Worker {i} Started on gpu {gpu_ids[i//num_workers]}, responsible for {len(index_groups)} images")
        
        for _ in range(num_gpus*num_workers):
            styleList += styledQueue.get()
            
        for process in workers:
            process.join()
    else:
        print("==>" + " No worker Started, main thread responsible for {} images".format(datasetSize))
        worker(0, dataset, target, style_dir, styles, list(range(datasetSize)), logger, styledQueue)
        styleList += styledQueue.get()
    
    cocoAndStyledAnnotations, styledOnlyAnnotations = createDatasets(dataset.coco.dataset, styles, styleList)
        
    for style in styles:
        assert(len(cocoAndStyledAnnotations[style]["images"]) == len(dataset.coco.dataset["images"])*2 == len(styledOnlyAnnotations[style]["images"])*2)
        assert(len(cocoAndStyledAnnotations[style]["annotations"]) == len(dataset.coco.dataset["annotations"])*2 == len(styledOnlyAnnotations[style]["annotations"])*2)
    
        with open(os.path.join(coco, "annotations", f"person_keypoints_coco_and_adain_{target}_{style}.json"), 'w') as file:
            json.dump(cocoAndStyledAnnotations[style], file)
        with open(os.path.join(coco, "annotations", f"person_keypoints_adain_{target}_{style}.json"), 'w') as file:
            json.dump(styledOnlyAnnotations[style], file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco', type=str, required=True, help='Path to the coco dataset')
    parser.add_argument('--style_dir', type=str, required=True, help='Directory with style datasets')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='GPUs to use')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use')
    parser.add_argument('--styles', type=str, nargs='+', default=[], help='List of styles to transform to')
    parser.add_argument('--log', type=str, default="", help='Path to log dir')
    args = parser.parse_args()
     
    if args.log != "":
        logging.basicConfig(
            filename=os.path.join(args.log, f"augmentation_adain_{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')}.log"),
            format='%(asctime)-15s %(message)s'
        )
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)
    else:
        logger = None
    
    trainPath = os.path.join(args.coco, "images", "train_adain")
    valPath = os.path.join(args.coco, "images", "val_adain")
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
    if not os.path.exists(valPath):
        os.makedirs(valPath)
        
    augment(args.coco, "train2017", "train_adain", args.style_dir, args.styles, args.gpu_ids, args.num_workers, logger)
    augment(args.coco, "val2017", "val_adain", args.style_dir, args.styles, args.gpu_ids, args.num_workers, logger)