import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
from torchvision.transforms import transforms
from utils import isArrayLike

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
    
    def validate(self, gpuIds, data_loader, direction):
        gpuIds = gpuIds if isArrayLike(gpuIds) else [gpuIds]

        predictions = []
        pbar = tqdm(total=len(data_loader.dataset)) if self.verbose else None
        for i, (images, annotations) in enumerate(data_loader):     
            image = images[0].float() / 255
            image = torch.permute(image, (2, 0, 1))
            image = transforms.Resize((256, 256))(image)
            image = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)
            image = torch.stack((image,))
            image = image.to("cuda")

            if direction == self.PHOTOGRAPHIC_TO_ARTISTIC:
                image = self.styleTransformer.artisticToPhotographic(image)
            else:    
                image = self.styleTransformer.photographicToArtistic(image)
            image = image[0] * 0.5 + 0.5
            image = image.detach().cpu().numpy()
            image = image.transpose(1, 2, 0) * 255
            if self.verbose:
                self.styleTransformer.visualize(image, f"style_transfer_{int(annotations[0]['image_id'])}")
            image_resized, final_heatmaps, final_results, scores = self.poseEstimator.infer(gpuIds, image)

            if self.verbose:
                self.poseEstimator.visualize(image_resized[0], final_heatmaps[0], f"pose_estimation_{int(annotations[0]['image_id'])}")
                pbar.update()

            for idx in range(len(final_results)):
                predictions.append({
                    "keypoints": final_results[idx][:,:3].reshape(-1,).astype(float).tolist(),
                    "image_id": int(annotations[0]["image_id"]),
                    "score": float(scores[idx]),
                    "category_id": 1
                })

        if self.verbose:
            pbar.close()
            
        return predictions