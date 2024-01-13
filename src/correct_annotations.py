import json
import os
import argparse
from multiprocessing import Process

def convert(worker_no, file_path):
    print(f"Worker {worker_no} started")
    with open(file_path, "r+") as file:
        annotationsJSON = json.load(file)
        images = []
        for image in annotationsJSON["images"]:
            if int(image["id"]) >= 1000000000000:
                image["file_name"] = os.path.join("train_corrected", image["file_name"])
            else:
                image["file_name"] = os.path.join("train2017", image["file_name"])
            images.append(image)
        annotationsJSON["images"] = images
        file.seek(0)
        file.truncate()
        file.write(json.dumps(annotationsJSON, indent="\t"))
        file.close()
        
    print(f"Worker {worker_no} stopped")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_dir', type=str, default="", help='Directory of the annotations')
    parser_args = parser.parse_args()
    
    annotation_folder = parser_args.annotation_dir
    files = os.listdir(annotation_folder)
    workers = []

    for idx, file_name in enumerate(files):
        process = Process(
            target = convert,
            args = (idx, os.path.join(annotation_folder, file_name))
        )
        process.start()
        workers.append(process)
    
    for process in workers:
        process.join()