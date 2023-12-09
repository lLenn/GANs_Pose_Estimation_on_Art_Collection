import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
from torchvision.transforms import transforms
from utils import isArrayLike

class ArtPose:
    def __init__(self, poseEstimator, styleTransformer, verbose=False):
        self.poseEstimator = poseEstimator
        self.styleTransformer = styleTransformer
        self.verbose = verbose
        
    def loadModel(self, isTrain=False):
        self.poseEstimator.loadModel(isTrain)
        self.styleTransformer.loadModel()
    
    def train():
        pass
    
    def validate(self, gpuIds, dataset, indices, logger):
        gpuIds = gpuIds if isArrayLike(gpuIds) else [gpuIds]
        
        sub_dataset = torch.utils.data.Subset(dataset, indices)
        data_loader = torch.utils.data.DataLoader(
            sub_dataset, sampler=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
        )
        predictions = []
        pbar = tqdm(total=len(sub_dataset)) if self.verbose else None
        for i, (images, annotations) in enumerate(data_loader):     
            image = images[0].float() / 255
            image = torch.permute(image, (2, 0, 1))
            image = transforms.Resize((256, 256))(image)
            image = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)
            image = torch.stack((image,))
            image = image.to("cuda")

            image = self.styleTransformer.transformFromArtisticToPhotographic(image)
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