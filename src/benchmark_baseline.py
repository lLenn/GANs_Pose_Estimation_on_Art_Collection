import json
import os
import itertools
import copy
from collections import OrderedDict
from utils import Logger
from multiprocessing import Process, Queue
from pose_estimation.networks.ArtPose import ArtPose
from available_algorithms import createDatasetIterator, createPoseEstimatorIterator, createStyleTransformerIterator


# This code will establish the base line for pose estimation done on pose estimation datasets that have been transformed with a style transfer algorithm.

NO_WORKERS = 4

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
 
def worker(gpuIds, dataset, indices, poseEstimator, styleTransformer, logger, finalOutputDir, predictionQueue):
    artPose = ArtPose(poseEstimator, styleTransformer, True)
    predictions = artPose.validate(gpuIds, dataset, indices, logger)    
    predictionQueue.put_nowait(predictions)
    
def benchmark(dataset, poseEstimator, styleTransformer, logger):
    datasetSize = len(dataset)
    predictionQueue = Queue(100)
    workers = []
    for i in range(NO_WORKERS):
        index_groups = list(range(i, datasetSize, NO_WORKERS))
        process = Process(
            target = worker,
            args = (
                0, dataset, index_groups, copy.deepcopy(poseEstimator), copy.deepcopy(styleTransformer), logger, "./.output", predictionQueue
            )
        )
        process.start()
        workers.append(process)
        logger.info("==>" + " Worker {} Started, responsible for {} images".format(i, len(index_groups)))
    
    allPredictions = []
    for _ in range(NO_WORKERS):
        allPredictions += predictionQueue.get()
    
    for process in workers:
        process.join()
        
    resultFolder = ".output/results"
    if not os.path.exists(resultFolder):
        os.makedirs(resultFolder)
    resultFile = os.path.join(resultFolder, f"keypoints_SWAHR_results.json")

    json.dump(allPredictions, open(resultFile, 'w'))

    info_str = dataset.keypointEvaluation(resultFile)
    name_values = OrderedDict(info_str)
    
    if isinstance(name_values, list):
        for name_value in name_values:
            printNameValue(logger, name_value, "SWAHR")
    else:
        printNameValue(logger, name_values, "SWAHR")

if __name__ == "__main__":
    logger = Logger("./.output/logs", "validate")
    
    for dataset, poseEstimator, styleTransformer in itertools.product(createDatasetIterator(), createPoseEstimatorIterator(), createStyleTransformerIterator()):
        benchmark(dataset, poseEstimator, styleTransformer, logger)