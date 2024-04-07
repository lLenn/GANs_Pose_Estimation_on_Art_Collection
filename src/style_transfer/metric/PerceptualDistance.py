# BASED ON: https://github.com/Tee0125/pytorch-fast-style-transfer/blob/master/losses/style_transfer_loss.py

import torch
import numpy as np
from torch import nn
from torch.functional import F
from torchvision.models import vgg19    
from itertools import chain
    
class PerceptualDistance():
    def __init__(self, rank):
        
        backbone = self.init_backbone(rank)
        self.backbone_content = backbone[0]
        self.backbone_style = backbone[1]
        
        self.device = torch.device(f"cuda:{rank}")

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(self.device)
        
        self.distance = []

    def process_images(self, content_images, style_image, generated_images):
        style_image = style_image.unsqueeze(0)
        
        content_images = self.normalize_image(content_images)
        style_image = self.normalize_image(style_image)
        generated_images = self.normalize_image(generated_images)

        
        loss = []
        loss.append(self.get_content_loss(content_images, generated_images))
        loss.append(self.get_style_loss(style_image, generated_images))
        self.distance.append(sum(loss).item())

    def get_perceptual_distance(self, rank, world_size):        
        if world_size == 1:
            return np.mean(self.distance)
        
        distance = torch.tensor(self.distance)
        gather_list = [None] * world_size
        torch.distributed.all_gather_object(gather_list, distance)

        if rank == 0:
            gathered_distances = torch.tensor(gather_list).view(1, -1)
            return torch.mean(gathered_distances).item()

    def get_content_loss(self, content_images, generated_images):
        content_images = self.backbone_content(content_images)
        generated_images = self.backbone_content(generated_images)

        return F.mse_loss(content_images, generated_images)

    def get_style_loss(self, style_image, generated_images):
        n = generated_images.size(0)

        style_features = self.get_style_features(style_image)
        generated_features = self.get_style_features(generated_images)
        
        l_style = 0
        for style_feature, generated_feature in zip(style_features, generated_features):
            l_style += F.mse_loss(generated_feature, style_feature.expand_as(generated_feature), reduction='sum')

        return l_style / n

    def get_style_features(self, images):
        features = []

        for layer in self.backbone_style:
            images = layer(images)
            features.append(self.gram_matrix(images))

        return features
    
    def normalize_image(self, images):
        return (images - self.mean) / self.std
        
    @staticmethod
    def init_backbone(rank):
        vgg = vgg19(pretrained=True)
        features = vgg.features[0:30]

        # freeze parameters
        features.eval()

        for p in features.parameters():
            p.requires_grad = False

        features = features.cuda(torch.device(f"cuda:{rank}"))

        # relu4_2
        backbone_content = features[0:23]

        # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
        backbone_style = [features[0:2],
                            features[2:7],
                            features[7:12],
                            features[12:21],
                            features[21:30]]

        backbone_style = nn.ModuleList(backbone_style)

        return backbone_content, backbone_style
    
    @staticmethod
    def gram_matrix(images):
        n = images.numel() / images.size(0)

        images = images.reshape(images.size(0), images.size(1), -1)
        images = torch.bmm(images, images.transpose(1, 2))

        return images / n
    