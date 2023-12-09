import os, cv2, torch
from glob import glob
from UGATITLib.UGATIT import UGATIT as UGATITNetwork

class UGATIT():
    def __init__(self, config):
        self.config = config
        self.model = UGATITNetwork(config)
        self.model.build_model()
        
    def loadModel(self, path=None):
        if path is None:
            params = torch.load(self.config.model_path)
        else:
            params = torch.load(path)
        self.model.genA2B.load_state_dict(params['genA2B'])
        self.model.genB2A.load_state_dict(params['genB2A'])
        self.model.disGA.load_state_dict(params['disGA'])
        self.model.disGB.load_state_dict(params['disGB'])
        self.model.disLA.load_state_dict(params['disLA'])
        self.model.disLB.load_state_dict(params['disLB'])
    
    def transformFromPhotographicToArtistic(self, image):
        self.model.genA2B.eval()
        image, _, _ = self.model.genA2B(image)
        return image
    
    def transformFromArtisticToPhotographic(self, image):
        self.model.genB2A.eval()
        image, _, _ = self.model.genB2A(image)
        return image
        
    def train(self):
        self.model.train()
        
    def print(self):
        print("Unsupervised generative attentional network with adaptive layer-instance for image-to-image translation")
        
    def visualize(self, image, name):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.model.result_dir, f"{name}.png"), image)