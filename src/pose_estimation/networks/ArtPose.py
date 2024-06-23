import torch
import numpy as np
import cv2
import os
from colour import Color
from tqdm import tqdm
from torchvision.transforms import transforms
from utils import isArrayLike
from pose_estimation.datasets import COCOSubset
from mmpose.datasets.datasets.body.coco_dataset import CocoDataset 

class ArtPose:
    PHOTOGRAPHIC_TO_ARTISTIC = "photographicToArtistic"
    ARTISTIC_TO_PHOTOGRAPHIC = "artisticToPhotographic"
    
    def __init__(self, poseEstimator, styleTransformer, verbose=False):
        self.poseEstimator = poseEstimator
        self.styleTransformer = styleTransformer
        self.verbose = verbose
        
    def loadModel(self, isTrain=False):
        self.poseEstimator.loadModel(isTrain)
        self.styleTransformer.loadModel()
    
    def train():
        pass
    
    def validate(self, rank, world_size, data_loader, direction, directory, metric_hook = None, resize = False, normalize = True):
        predictions = []
        pbar = tqdm(total=len(data_loader.dataset)) if self.verbose else None
        for i, (images, annotations) in enumerate(data_loader):
            image = images[0].float() / 255
            image = torch.permute(image, (2, 0, 1))
            bbox = [torch.tensor(annotation["bbox"]).numpy() for annotation in annotations]
            _, h, w = image.shape
            rh = h
            if not resize == False and (w > resize or h > resize):
                image = transforms.Resize(resize)(image)
                _, rh, _ = image.shape
                bbox = [torch.tensor([coor*rh/h for coor in annotation["bbox"]]).numpy() for annotation in annotations]
            if normalize:
                image = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)
            image = torch.stack((image,))
            image = image.to("cuda")

            if direction == self.PHOTOGRAPHIC_TO_ARTISTIC:
                image = self.styleTransformer.photographicToArtistic(image)
            else:    
                image = self.styleTransformer.artisticToPhotographic(image)
            image = image[0]
            if normalize:
                image = image * 0.5 + 0.5
            image = image.detach().cpu().numpy()
            image = image.transpose(1, 2, 0) * 255
            
            # add bbox for ViTPose
            
            if self.verbose:
                ArtPose.visualizeStyleTransfer(image, directory, f"style_transfer_{int(annotations[0]['image_id'])}")
            image_resized, final_heatmaps, final_results, scores = self.poseEstimator.infer(rank, world_size, image, bbox)
                
            if self.verbose:
                ArtPose.visualizePoseEstimation(image, np.delete(final_results, -1, 2).reshape(len(final_results), -1).astype(float).tolist(), scores, directory, annotations[0]["image_id"].item(), False)
                pbar.update()

            for idx in range(len(final_results)):
                keypoints = final_results[idx][:,:3].reshape(-1,).astype(float).tolist()
                for i, keypoint in enumerate(keypoints):
                    if i % 3 < 2:
                        keypoints[i] = keypoints[i]*h/rh
                predictions.append({
                    "keypoints": keypoints,
                    "image_id": int(annotations[0]["image_id"]),
                    "score": float(scores[idx]),
                    "category_id": 1
                })
            
            if metric_hook is not None:
                metric_hook(predictions[:-len(final_results)])

        if self.verbose:
            pbar.close()
            
        return predictions

    def visualizeStyleTransfer(image, directory, name):
        cv2.imwrite(os.path.join(directory, f"{name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    def visualizePoseEstimation(image, predictions, scores, directory, name, show_score):
        image = image.copy()
        def convertToIntTuple(arr):
            return int(arr[0]), int(arr[1])
        for i, prediction in enumerate(predictions):
            overall_color:Color = ArtPose.getColor(scores[i])
            for edge in COCOSubset.META_INFO["skeleton"]:
                indexP1 = (edge[0]-1)*3
                indexP2 = (edge[1]-1)*3
                edge_color = tuple(np.array(overall_color.get_rgb())*255) if show_score is True else tuple(edge[2])
                cv2.line(image, convertToIntTuple(prediction[indexP1:indexP1+2]), convertToIntTuple(prediction[indexP2:indexP2+2]), edge_color, 5)
            
            for predictionIndex in range(0, len(prediction), 3):
                keypoint_color:Color = tuple(ArtPose.getColor(prediction[predictionIndex+2]).get_rgb())*255 if show_score else tuple(COCOSubset.META_INFO["keypoints"][int(predictionIndex/3)][1])
                cv2.circle(image, convertToIntTuple(prediction[predictionIndex:predictionIndex+2]), 7, keypoint_color, thickness=-1)
                cv2.circle(image, convertToIntTuple(prediction[predictionIndex:predictionIndex+2]), 7, (255, 255, 255))
                
        cv2.imwrite(os.path.join(directory, f"{name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    def getColor(percentage):
        hue = percentage * 0.3
        return Color(hsl=(hue, 1, 0.5))