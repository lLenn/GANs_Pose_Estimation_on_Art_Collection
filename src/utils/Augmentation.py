import os
import cv2
import copy
from datetime import datetime

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
    dirname, basename = os.path.split(metadata["file_name"])
    root_file_name, ext = os.path.splitext(basename)
    
    copyMetadata = copy.deepcopy(metadata)
    copyMetadata["id"] = metadata["id"] + id_addition
    copyMetadata["file_name"] = os.path.join(dirname, f"{int(root_file_name) + id_addition}{ext}")
    return copyMetadata
    
def copyAnnotations(annotation, id_addition):
    copyAnnotations = copy.deepcopy(annotation)
    for copiedAnnotation in copyAnnotations:
        copiedAnnotation["id"] = copiedAnnotation["id"] + id_addition
        copiedAnnotation["image_id"] = copiedAnnotation["image_id"] + id_addition
    return copyAnnotations

def saveImage(path, image):
    image = image.detach().cpu().numpy()
    image = image.transpose(1,2,0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image * 255.0)
    
def createDatasets(dataset, styles, styleList):
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
        cocoAndStyledAnnotations[style] = copyDataset(dataset)
        styledOnlyAnnotations[style] = copyDataset(dataset, True)
        if style != "mixed":
            countStyle = len(dataset["images"])//lenStyles
            if styleIndex < len(dataset["images"])%lenStyles:
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
    
    return cocoAndStyledAnnotations, styledOnlyAnnotations