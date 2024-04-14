import os
import json
from collections import OrderedDict
from pycocotools.cocoeval import COCOeval
from torch import distributed

class AveragePrecision:
    def __init__(self, coco, results_dir, results_prefix):
        self.coco = coco
        self.result_file = os.path.join(results_dir, f"{results_prefix}_keypoint_results.json")
        self.firstProcess = True            
        
    def process_predictions(self, rank, world_size, predictions):
        if self.firstProcess:
            if rank == 0:
                with open(self.result_file, "w") as file:
                    file.write("[")
            if world_size > 1:
                distributed.barrier()
        for i in range(world_size):
            if world_size > 1:
                distributed.barrier()
            if i == rank:
                with open(self.result_file, "a") as file:
                    for prediction in predictions:
                        if self.firstProcess:
                            self.firstProcess = False
                            if rank == 0:
                                file.write(json.dumps(prediction))
                            elif rank != 0:
                                file.write(f",{json.dumps(prediction)}")
                        else:
                            file.write(f",{json.dumps(prediction)}")
        
    def get_average_precision(self, rank):
        if rank == 0:
            with open(self.result_file, "a") as file:
                file.write("]")
            
            coco = None
            try:
                coco = self.coco.loadRes(self.result_file)
            except:
                coco = self.coco.loadResults(self.result_file)
                
            humanArtEvaluation = COCOeval(self.coco, coco, 'keypoints')
            humanArtEvaluation.evaluate()
            humanArtEvaluation.accumulate()
            humanArtEvaluation.summarize()
            stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

            info_str = {}
            for ind, name in enumerate(stats_names):
                info_str[name] = humanArtEvaluation.stats[ind]

            return info_str
        else:
            return 0
        
    