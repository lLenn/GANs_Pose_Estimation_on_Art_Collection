import json
import os
import argparse
import re
from multiprocessing import Process

# Adds the correct folder to the filename
# ViTPose and ArtPose use filename relative from coco root

def convert(worker_no, file_path):
    print(f"Worker {worker_no} started")
    with open(file_path, "r+") as file:
        prefix = re.search(r"(_val|_train)", os.path.basename(file_path)).group(0)[1:]
        annotationsJSON = json.load(file)
        images = []
        for image in annotationsJSON["images"]:
            id = int(image["id"])
            if id >= 7000000000000 and id < 8000000000000:
                image["file_name"] = os.path.join(f"{prefix}_corrected", image["file_name"])
            elif id >= 8000000000000:
                image["file_name"] = os.path.join(f"{prefix}_adain", image["file_name"])
            else:
                image["file_name"] = os.path.join(f"{prefix}2017", image["file_name"])
            images.append(image)
        annotationsJSON["images"] = images
        file.seek(0)
        file.truncate()
        file.write(json.dumps(annotationsJSON))
        file.close()
        
    print(f"Worker {worker_no} stopped")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_dir', type=str, default="", help='Directory of the annotations')
    parser_args = parser.parse_args()
    
    annotations_folder = parser_args.annotations_dir
    files = os.listdir(annotations_folder)
    workers = []

    for idx, file_name in enumerate(files):
        process = Process(
            target = convert,
            args = (idx, os.path.join(annotations_folder, file_name))
        )
        process.start()
        workers.append(process)
    
    for process in workers:
        process.join()