import numpy as np
from pose_estimation.datasets import COCOSubset

MAX_DETECTIONS = 20

class ObjectKeypointSimilarity:
    def __init__(self, ground_truths, predictions):
        self.ground_truths = ground_truths
        self.predictions = predictions
        
    def calculateOKS(self):
        sorted_indices = np.argsort([-prediction['score'] for prediction in self.predictions], kind='mergesort')
        predictions = [self.predictions[i] for i in sorted_indices]
        if len(predictions) > MAX_DETECTIONS:
            predictions = predictions[0:MAX_DETECTIONS]
        if len(self.ground_truths) == 0 or len(predictions) == 0:
            return []
        self.ious = np.zeros((len(predictions), len(self.ground_truths)))
        sigmas = np.array(COCOSubset.META_INFO["sigmas"])
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, ground_truth in enumerate(self.ground_truths):
            # create bounds for ignore regions(double the ground_truth bbox)
            g = np.array(ground_truth['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = ground_truth['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(predictions):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (ground_truth['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                self.ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    
    def bestPredictions(self):
        if len(self.ground_truths) == 0 or len(self.predictions) == 0:
            return None

        prediction_indices = np.argsort([-prediction['score'] for prediction in self.predictions], kind='mergesort')
        predictions = [self.predictions[i] for i in prediction_indices[0:MAX_DETECTIONS]]
        matched_ground_truth = [0]*len(self.ground_truths)
        best_predictions = []
        
        if not len(self.ious)==0:
            for prediction_index, prediction in enumerate(predictions):
                iou = 0
                match = -1
                for ground_truth_index, ground_truth in enumerate(self.ground_truths):
                    # continue to next ground truth unless better match made
                    if self.ious[prediction_index,ground_truth_index] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou=self.ious[prediction_index,ground_truth_index]
                    match=ground_truth_index
                if matched_ground_truth[match] != 1:
                    matched_ground_truth[match] = 1
                    best_predictions.append(prediction)
        
        return best_predictions
        