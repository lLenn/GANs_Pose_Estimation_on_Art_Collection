from StarGAN.metrics.lpips import LPIPS
import numpy as np
import torch

class LearnedPerceptualImagePatchSimilarity:
    def __init__(self, size=512):
        self.size = size
        self.generated_images = np.empty((0,3,size,size))
        self.real_images = np.empty((0,3,size,size))
        
    def process_generated_images(self, images):
        self.generated_images = np.append(self.generated_images, images.cpu().detach().numpy(), axis=0)
        
    def process_real_images(self, images):
        self.real_images = np.append(self.real_images, images.cpu().detach().numpy(), axis=0)
    
    def get_lpips(self, rank, world_size):
        generated_images = np.empty((0,3,self.size,self.size))
        real_images = np.empty((0,3,self.size,self.size))
        gather_generated = [None] * world_size
        gather_real = [None] * world_size
        if world_size > 1:
            torch.distributed.all_gather_object(gather_generated, torch.tensor(self.generated_images))
            torch.distributed.all_gather_object(gather_real, torch.tensor(self.real_images))
        else:
            gather_generated = [torch.tensor(self.generated_images)]
            gather_real = [torch.tensor(self.real_images)]
        
        if rank == 0:
            for i in range(world_size):
                generated_images = np.append(generated_images, gather_generated[i].cpu().detach().numpy(), axis=0)
                real_images = np.append(real_images, gather_real[i].cpu().detach().numpy(), axis=0)

            lpips = LPIPS().eval().cuda()
            lpips_values_similarity = []
            lpips_values_variation = []

            generated_images = torch.tensor(generated_images).type(torch.FloatTensor).cuda()
            real_images = torch.tensor(real_images).type(torch.FloatTensor).cuda()

            for i in range(len(generated_images)):
                for j in range(len(real_images)):
                    lpips_values_similarity.append(lpips(generated_images[i], real_images[j]).cpu().detach().numpy())
                for j in range(i+1, len(generated_images)):
                    lpips_values_variation.append(lpips(generated_images[i], generated_images[j]).cpu().detach().numpy())
            return np.mean(lpips_values_similarity), np.mean(lpips_values_variation)
        else:
            return 0, 0
    