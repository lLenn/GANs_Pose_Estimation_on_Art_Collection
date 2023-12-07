import json
from os.path import abspath

with open("../../datasets/human-art/annotations/test_humanart_oil_painting.json", "r+") as file:
    annotationsJSON = json.load(file)
    annotationsJSON["info"]["description"] = "Custom Human-Art Dataset for testing"
    images = []
    for image in annotationsJSON["images"]:
        if int(image["id"]) >= 6000000000000 and int(image["id"]) <= 6000000000100:
            images.append(image)
    annotationsJSON["images"] = images
    annotations = []
    for annotation in annotationsJSON["annotations"]:
        if int(annotation["image_id"]) >= 6000000000000 and int(annotation["image_id"]) <= 6000000000100:
            annotations.append(annotation)
    annotationsJSON["annotations"] = annotations
    file.seek(0)
    file.truncate()
    file.write(json.dumps(annotationsJSON, indent="\t"))