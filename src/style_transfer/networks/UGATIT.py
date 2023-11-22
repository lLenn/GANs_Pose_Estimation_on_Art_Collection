import os
import cv2
from glob import glob
from UGATITLib.UGATIT import UGATIT as UGATITNetwork

class UGATIT():
    def __init__(self, config):
        self.model = UGATITNetwork(config)
        self.model.build_model()
    
    def train(self):
        self.model.train()
        
    def loadModel(self, path):
        self.model.load(path, 1000)
    
    def transformFromPhotographicToArtistic(self, image):
        self.model.genA2B.eval()
        image, _, _ = self.model.genA2B(image)
        return image
    
    def transformFromArtisticToPhotographic(self, image):
        self.model.genB2A.eval()
        image, _, _ = self.model.genB2A(image)
        return image
        
    def print(self):
        print("Unsupervised generative attentional network with adaptive layer-instance for image-to-image translation")
        
    def visualize(self, image, name):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.model.result_dir, f"{name}.png"), image)