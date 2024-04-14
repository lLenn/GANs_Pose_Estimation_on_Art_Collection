import json
import argparse
import random
import os
from multiprocessing import Process

# Adds the correct folder to the filename
# ViTPose and ArtPose use filename relative from coco root

def subset(file_path, subset):
    selected = []
    annotationsJSON = None
    with open(file_path, "r+") as file:
        annotationsJSON = json.load(file)
    
    dirname, filename = os.path.split(file_path)    
    with open(os.path.join(dirname, f"subset_{filename}"), "w") as file:
        annotationsJSON["info"]["description"] = "Subset COCO Dataset for testing"
        dataset_len = len(annotationsJSON["images"])
        images = []
        for _ in range(subset):
            has_annotations = False
            index = random.randint(0, dataset_len-1)
            for annotation in annotationsJSON["annotations"]:
                if int(annotation["image_id"]) == int(annotationsJSON["images"][index]["id"]) and int(annotation["num_keypoints"]) > 0:
                    has_annotations = True
                    break
            while int(annotationsJSON["images"][index]["id"]) in selected or not has_annotations:
                has_annotations = False
                index = random.randint(0, dataset_len)
                for annotation in annotationsJSON["annotations"]:
                    if int(annotation["image_id"]) == int(annotationsJSON["images"][index]["id"]) and int(annotation["num_keypoints"]) > 0:
                        has_annotations = True
                        break
            selected.append(int(annotationsJSON["images"][index]["id"]))
            images.append(annotationsJSON["images"][index])
            
        annotations = []
        for annotation in annotationsJSON["annotations"]:
            if int(annotation["image_id"]) in selected:
                annotations.append(annotation)
                
        annotationsJSON["images"] = images
        annotationsJSON["annotations"] = annotations
        
        file.write(json.dumps(annotationsJSON))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="", help='File to subset')
    parser.add_argument('--subset', type=int, default="", help='Number in subset')
    parser_args = parser.parse_args()
    
    subset(parser_args.file, parser_args.subset)