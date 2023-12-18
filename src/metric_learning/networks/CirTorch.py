import pickle
import torch
import numpy as np
from torchvision import transforms
from utils.ObjectHelpers import isArrayLike

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime

class CirTorch():
    def __init__(self, config):
        self.config = config
        self.network = None
        self.transform = None
        self.multiscale = ms = list(config.multiscale)
        
    def load(self):
        # loading network
        print(">> Loading network")
        state = torch.load(self.config.network_path)
        # parsing net params from meta
        # architecture, pooling, mean, std required
        # the rest has default values, in case that is doesnt exist
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get('local_whitening', False)
        net_params['regional'] = state['meta'].get('regional', False)
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False
        # network initialization
        self.network = init_network(net_params)
        self.network.load_state_dict(state['state_dict'])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.network.meta['mean'],
                std=self.network.meta['std']
            )
        ])
    
    def saveDatabase(self, path):
        if self.database is not None:
            with open(path, 'wb') as database:
                pickle.dump([self.images, self.database], database)
            
    def loadDatabase(self, path):
        with open(path, 'rb') as database:
            dump = pickle.load(database)
            self.images = dump[0]
            self.database = dump[1]
    
    def createDatabase(self, images):
        self.network.cuda()
        self.network.eval()
        self.images = images
        self.database = extract_vectors(self.network, images, self.config.image_size, self.transform, ms=self.multiscale).numpy()
        
    def query(self, queryImages):
        if isinstance(queryImages, str):
            queryImages = [queryImages]
        self.network.cuda()
        self.network.eval()
        queryVectors = extract_vectors(self.network, queryImages, self.config.image_size, self.transform, ms=self.multiscale)
        queryVectors = queryVectors.numpy()
        scores = np.dot(self.database.T, queryVectors)
        rank = np.argsort(-scores, axis=0)
        return np.concatenate((rank, np.reshape(scores[rank], (scores.shape[0],1))), axis=1)
        