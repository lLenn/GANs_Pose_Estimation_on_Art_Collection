import json
import os
import logging
import copy
import argparse
import cv2
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torchvision.transforms import transforms
from multiprocessing import Process, Queue
from SWAHR.dataset.COCODataset import CocoDataset
from style_transfer.networks import CycleGAN, CycleGANConfig

ID_ADDITION = 7000000000000
ID_SUB_ADDITION = 100000000000

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
    
def copyDataset(cocoDataset, removeData=False):
    dataset = copy.deepcopy(cocoDataset)
    dataset["info"]["description"] = "ArtPose Dataset"
    dataset["info"]["url"] = "https://github.com/lLenn/GANs_Pose_Estimation_on_Art_Collection.git"
    dataset["info"]["version"] = "1.0"
    dataset["info"]["year"] = str(datetime.year)
    dataset["info"]["contributor"] = "University Ghent"
    dataset["info"]["date_created"] = datetime.strftime(datetime.now(), "%Y/%m/%d")
    
    if removeData:
        dataset["images"] = []
        dataset["annotations"] = []
        
    return dataset

def copyMetadata(metadata, id_addition):
    root_file_name, ext = os.path.splitext(metadata["file_name"])
    
    copyMetadata = copy.deepcopy(metadata)
    copyMetadata["id"] = metadata["id"] + id_addition
    copyMetadata["file_name"] = f"{int(root_file_name) + id_addition}{ext}"
    return copyMetadata
    
def copyAnnotations(annotation, id_addition):
    copyAnnotations = copy.deepcopy(annotation)
    for copiedAnnotation in copyAnnotations:
        copiedAnnotation["id"] = copiedAnnotation["id"] + id_addition
        copiedAnnotation["image_id"] = copiedAnnotation["image_id"] + id_addition
    return copyAnnotations

def saveImage(path, image):
    image = image * 0.5 + 0.5
    image = image.detach().cpu().numpy()
    image = image.transpose(1,2,0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image * 255.0)

def worker(gpuId, dataset, target, models, styles, workerIndices, logger, styledQueue):
    # Initiate the CycleGAN model with the proper configuration
    config = CycleGANConfig.create()
    config.defrost()
    config.isTrain = False
    config.freeze()
    style_transfer = CycleGAN(config)
    
    # Each model found at the given path will be iterated
    model_list = [file for file in os.listdir(models) if os.path.isdir(os.path.join(models, file))]
    indicesSize = len(workerIndices)
    
    # Array that will be passed back to the main process
    style_list = []
    
    # Transform image to proper input for cyclegan model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    pbar = tqdm(total=indicesSize*len(model_list)) if logger is not None else None
    # Each model is iterated and the workerIndices are split equally so 
    for modelIndex, model in enumerate(model_list):
        if model not in styles:
            if logger is not None:
                pbar.update(indicesSize)
            continue
        id_addition = ID_ADDITION + ID_SUB_ADDITION * modelIndex
        style_transfer.loadModel(os.path.join(models, model), withName=False)
        for imageIndex in workerIndices:
            img_id = dataset.ids[imageIndex]
            metadata = dataset.coco.loadImgs(img_id)[0]
            file_path = os.path.join(dataset.root, 'images', dataset.dataset, metadata['file_name'])
            image = cv2.imread(file_path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            image = transform(image).unsqueeze(0)
            styledImage = style_transfer.photographicToArtistic(image, gpuId).squeeze()
            copiedMetadata = copyMetadata(metadata, id_addition)
            copiedAnnotations = copyAnnotations(dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=metadata["id"])), id_addition)
            file_name = os.path.join(dataset.root, 'images', target, copiedMetadata["file_name"])
            saveImage(file_name, styledImage)
            style_list.append({
                "style": model,
                "metadata": copiedMetadata,
                "annotations": copiedAnnotations
            })            
            if logger is not None:
                pbar.update()
    
    if logger is not None:
        pbar.close()
    styledQueue.put_nowait(style_list)
    
def augment(coco, source, target, models, styles, gpu_ids, num_workers, logger):
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
                    i//num_workers, dataset, target, models, styles, index_groups, logger, styledQueue
                )
            )
            process.start()
            workers.append(process)
            logger.info("==>" + " Worker {} Started, responsible for {} images".format(i, len(index_groups)))
        
        for _ in range(num_gpus*num_workers):
            styleList += styledQueue.get()
            
        for process in workers:
            process.join()
    else:
        logger.info("==>" + " No worker Started, main thread responsible for {} images".format(datasetSize))
        worker(0, dataset, target, models, styles, list(range(datasetSize)), logger, styledQueue)
        styleList += styledQueue.get()
        
    # Copying the annotations to create our own datasets
    # For each style we keep track of the already added number, as well as the image name
    # The max number is determined by floor division and + 1 if style index is within the remainder
    cocoAndStyledAnnotations = dict()
    styledOnlyAnnotations = dict()
    addedMixed = dict()
    addedIds = []
    withMixed = "mixed" in styles
    lenStyles = len(styles) - (1 if withMixed else 0)
    styleIndex = 0
    for style in styles:
        cocoAndStyledAnnotations[style] = copyDataset(dataset.coco.dataset)
        styledOnlyAnnotations[style] = copyDataset(dataset.coco.dataset, True)
        if style != "mixed":
            countStyle = len(dataset.coco.dataset["images"])//lenStyles
            if styleIndex < len(dataset.coco.dataset["images"])%lenStyles:
                countStyle += 1
            addedMixed[style] = [0, countStyle]
            styleIndex += 1
    
    for item in styleList:
        id = int(str(item["metadata"]["id"])[2:])
        cocoAndStyledAnnotations[item["style"]]["images"].append(item["metadata"])
        cocoAndStyledAnnotations[item["style"]]["annotations"] += item["annotations"]
        styledOnlyAnnotations[item["style"]]["images"].append(item["metadata"])
        styledOnlyAnnotations[item["style"]]["annotations"] += item["annotations"]
    
        if withMixed:
            if addedMixed[item["style"]][0] < addedMixed[item["style"]][1] and id not in addedIds:
                styledOnlyAnnotations["mixed"]["images"].append(item["metadata"])
                styledOnlyAnnotations["mixed"]["annotations"] += item["annotations"]
                addedMixed[item["style"]][0] += 1
                addedIds.append(id)
    cocoAndStyledAnnotations["mixed"]["images"] += styledOnlyAnnotations["mixed"]["images"]
    cocoAndStyledAnnotations["mixed"]["annotations"] += styledOnlyAnnotations["mixed"]["annotations"]
        
    for style in styles:
        assert(len(cocoAndStyledAnnotations[style]["images"]) == len(dataset.coco.dataset["images"])*2 == len(styledOnlyAnnotations[style]["images"])*2)
        assert(len(cocoAndStyledAnnotations[style]["annotations"]) == len(dataset.coco.dataset["annotations"])*2 == len(styledOnlyAnnotations[style]["annotations"])*2)
    
        with open(os.path.join(coco, "annotations", f"person_keypoints_coco_and_styled_{target}_{style}.json"), 'w') as file:
            json.dump(cocoAndStyledAnnotations[style], file)
        with open(os.path.join(coco, "annotations", f"person_keypoints_styled_{target}_{style}.json"), 'w') as file:
            json.dump(styledOnlyAnnotations[style], file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco', type=str, required=True, help='Path to the coco dataset')
    parser.add_argument('--models', type=str, required=True, help='Directory with models to use for transfer')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='GPUs to use')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use')
    parser.add_argument('--styles', type=str, nargs='+', default=[], help='List of styles to transform to')
    parser.add_argument('--log', type=str, default="", help='Path to log dir')
    args = parser.parse_args()
     
    if args.log != "":
        logging.basicConfig(
            filename=os.path.join(args.log, f"augmentation_cyclegan_{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')}.log"),
            format='%(asctime)-15s %(message)s'
        )
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)
    else:
        logger = None
    
    trainPath = os.path.join(args.coco, "images", "train")
    valPath = os.path.join(args.coco, "images", "val")
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
    if not os.path.exists(valPath):
        os.makedirs(valPath)
        
    augment(args.coco, "train2017", "train", args.models, args.styles, args.gpu_ids, args.num_workers, logger)
    augment(args.coco, "val2017", "val", args.models, args.styles, args.gpu_ids, args.num_workers, logger)