import json
from os.path import abspath

ids = [6000000000209, 6000000000301, 6000000000782, 6000000000997, 6000000001690, 6000000002290, 6000000002693, 6000000003198, 6000000003528, 6000000000011, 6000000000084, 6000000000183, 6000000000290, 6000000000606, 6000000000710, 6000000000983, 6000000001732, 6000000002004, 6000000002219, 6000000002867, 6000000003414]

with open("../../Datasets/HumanArt/annotations/test_humanart_oil_painting.json", "r+") as file:
    annotationsJSON = json.load(file)
    annotationsJSON["info"]["description"] = "Custom Human-Art Dataset for testing"
    images = []
    for image in annotationsJSON["images"]:
        if int(image["id"]) in ids:
            images.append(image)
    annotationsJSON["images"] = images
    annotations = []
    for annotation in annotationsJSON["annotations"]:
        if int(annotation["image_id"]) in ids:
            annotations.append(annotation)
    annotationsJSON["annotations"] = annotations
    file.seek(0)
    file.truncate()
    file.write(json.dumps(annotationsJSON, indent="\t"))