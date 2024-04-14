import numpy as np
from itertools import chain
from torch import distributed
from mmpose.evaluation.functional import ( keypoint_pck_accuracy )

# Adapted from ViTPose: https://github.com/open-mmlab/mmengine/blob/main/mmengine/evaluator/metric.py
class PercentageCorrectKeypoints:
    def __init__(self, threshold = 0.05):
        self.results = []
        self.threshold = threshold
        
    def process_predictions(self, groundtruths, predictions):
        for i, prediction in enumerate(predictions):
            groundtruth = groundtruths[i]
            pred_coords = prediction['keypoints']
            gt_coords = groundtruth['keypoints']
            
            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords
            }

            bbox_size_ = np.max(groundtruth['bbox'][2:] - groundtruth['bbox'][:2])
            bbox_size = np.array([bbox_size_, bbox_size_]).reshape(-1, 2)
            result['bbox_size'] = bbox_size
                
            self.results.append(result)

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }
        
    def get_percentage_correct_keypoints(self, rank, world_size):
        results = [None] * world_size
        if world_size > 0:
            distributed.all_gather_object(results, self.results)
            results = [i for i in chain(results)]
        else:
            results = self.results
 
        pred_coords = np.concatenate([result['pred_coords'][:2] for result in results])
        gt_coords = np.concatenate([result['gt_coords'][:2] for result in results])
        mask = np.concatenate([result['gt_coords'][2:] for result in results])
        norm_size_bbox = np.concatenate([result['bbox_size'] for result in results])
        
        '''
        
        
        _, pck, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask, self.threshold, norm_size_bbox)
    distances = _calc_distances(pred, gt, mask, norm_factor)
     N, K, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    _mask[np.where((norm_size_bbox == 0).sum(1))[0], :] = False

    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    norm_factor[np.where(norm_factor <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1)
    return distances.T

    acc = np.array([_distance_acc(d, thr) for d in distances])
    
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0.0
    return acc, avg_acc, cnt
        
        return pck
        '''
        return None