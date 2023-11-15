import json
import os
from collections import OrderedDict
from pose_estimation.networks import SWAHR
from style_transfer.datasets import HumanArtDataset
from style_transfer.networks import UGATIT
from utils import Logger, Config
from utils.Programs import ValidateNetworkProgram
from multiprocessing import Process, Queue

# python .\tools\dist_valid.py --world_size 2 --cfg .\experiments\coco\higher_hrnet\w32_512_adam_lr1e-3.yaml TEST.MODEL_FILE .\models\pose_coco\pose_higher_hrnet_w32_512.pth
# This code will establish the base line for pose estimation done on pose estimation datasets that have been transformed with a style transfer algorithm.

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
 
def worker(gpuIds, dataset, indices, config, logger, finalOutputDir, predictionQueue):
    networkSWAHR = SWAHR(config)
    predictions = networkSWAHR.validate(gpuIds, dataset, indices, logger)    
    predictionQueue.put_nowait(predictions)
    
    '''
    networkUGATIT = UGATIT()
    networkUGATIT.print()
    networkUGATIT.train()
    '''

def main():
    program = ValidateNetworkProgram()
    
    config = Config.create("./study/pose_estimation/models/test.yaml", program.getArgument("opts"))
    Config.configureEnvironment(config)
    
    logger = Logger("./output/logs", "validate")
    logger.info(config)

    datasetHumanArt = HumanArtDataset("../../datasets/human-art", "oil_painting", "test", False)
    datasetSize = len(datasetHumanArt)
    predictionQueue = Queue(100)
    workers = []
    worldSize = program.getArgument("world_size")
    for i in range(worldSize):
        index_groups = list(range(i, datasetSize, worldSize))
        process = Process(
            target = worker,
            args = (
                0, datasetHumanArt, index_groups, config, logger, "./.output", predictionQueue
            )
        )
        process.start()
        workers.append(process)
        logger.info("==>" + " Worker {} Started, responsible for {} images".format(i, len(index_groups)))
    
    allPredictions = []
    for _ in range(program.getArgument("world_size")):
        allPredictions += predictionQueue.get()
    
    for process in workers:
        process.join()
        
    resultFolder = "./output/results"
    if not os.path.exists(resultFolder):
        os.makedirs(resultFolder)
    resultFile = os.path.join(resultFolder, f"keypoints_SWAHR_results.json")

    json.dump(allPredictions, open(resultFile, 'w'))

    info_str = datasetHumanArt.keypointEvaluation(resultFile)
    name_values = OrderedDict(info_str)
    
    if isinstance(name_values, list):
        for name_value in name_values:
            printNameValue(logger, name_value, "SWAHR")
    else:
        printNameValue(logger, name_values, "SWAHR")

if __name__ == "__main__":
    main()