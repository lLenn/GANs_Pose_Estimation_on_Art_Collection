import json
from os.path import abspath

with open("../../Datasets/custom/coco_annotations_small/annotations/person_keypoints_val2017.json", "r+") as file:
    annotationsJSON = json.load(file)
    annotationsJSON["info"]["description"] = "Custom Human-Art Dataset for testing"
    images = []
    for image in annotationsJSON["images"]:
        if int(image["id"]) in [872, 4134]:
            images.append(image)
    annotationsJSON["images"] = images
    annotations = []
    for annotation in annotationsJSON["annotations"]:
        if int(annotation["image_id"]) in [872, 4134]:
            annotations.append(annotation)
    annotationsJSON["annotations"] = annotations
    file.seek(0)
    file.truncate()
    file.write(json.dumps(annotationsJSON, indent="\t"))