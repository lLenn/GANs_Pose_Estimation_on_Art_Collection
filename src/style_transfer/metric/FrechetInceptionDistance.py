from pytorch_gan_metrics.utils import get_inception_feature
import numpy as np
import torch
from scipy import linalg
from itertools import chain

class FrechetInceptionDistance:
    def __init__(self):
        self.generated_acts = np.empty((0,2048))
        self.real_acts = np.empty((0,2048))
        
    def process_generated_images(self, images):
        acts, = get_inception_feature(images, dims=[2048], use_torch=False)
        self.generated_acts = np.append(self.generated_acts, acts, axis=0)
        
    def process_real_images(self, images):
        acts, = get_inception_feature(images, dims=[2048], use_torch=False)
        self.real_acts = np.append(self.real_acts, acts, axis=0)

    def frechet_distance(mu, cov, mu2, cov2):
        cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
        dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
        return np.real(dist)
    
    def get_frechet_inception_distance(self, rank, world_size):
        generated_acts = np.empty((0,2048))
        real_acts = np.empty((0,2048))
        gather_generated = [None] * world_size
        gather_real = [None] * world_size
        if world_size > 1:
            torch.distributed.all_gather_object(gather_generated, torch.tensor(self.generated_acts))
            torch.distributed.all_gather_object(gather_real, torch.tensor(self.real_acts))
        else:
            gather_generated = [torch.tensor(self.generated_acts)]
            gather_real = [torch.tensor(self.real_acts)]
        
        if rank == 0:
            for i in range(world_size):
                generated_acts = np.append(generated_acts, gather_generated[i].cpu().detach().numpy(), axis=0)
                real_acts = np.append(real_acts, gather_real[i].cpu().detach().numpy(), axis=0)

            mu_generated = np.mean(generated_acts, axis=0)
            cov_generated = np.cov(generated_acts, rowvar=False)
            
            mu_real = np.mean(real_acts, axis=0)
            cov_real = np.cov(real_acts, rowvar=False)
            
            return FrechetInceptionDistance.frechet_distance(mu_generated, cov_generated, mu_real, cov_real)
        else:
            return 0