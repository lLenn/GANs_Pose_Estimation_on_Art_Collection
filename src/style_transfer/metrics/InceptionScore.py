from pytorch_gan_metrics.utils import get_inception_feature
import numpy as np
import torch
from itertools import chain

class InceptionScore:
    def __init__(self, rank):
        self.device = torch.device(f"cuda:{rank}")
        self.probs = np.empty((0,1008))
        
    def process_generated_images(self, images):
        probs, = get_inception_feature(images, dims=[1008], use_torch=False, device=self.device)
        self.probs = np.append(self.probs, probs, axis=0)

    def get_inception_score(self, rank, world_size):
        scores = []
        kl = self.probs * (np.log(self.probs) - np.log(np.expand_dims(np.mean(self.probs, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
        
        if world_size == 1:
            gather_list = [torch.tensor(scores)]
            # return np.mean(scores), np.std(scores)
        else:
            scores = torch.tensor(scores)
            gather_list = [None] * world_size
            torch.distributed.all_gather_object(gather_list, scores)

        if rank == 0:
            gathered_scores = [i.cpu().detach().numpy() for i in chain.from_iterable(gather_list)]
            return np.mean(gathered_scores), np.std(gathered_scores)
        else:
            return 0, 0