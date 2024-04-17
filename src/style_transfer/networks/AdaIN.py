import torch
import torch.nn as nn
from AdaIN.net import vgg, decoder
from AdaIN.function import adaptive_instance_normalization, coral
from torchvision.transforms import transforms
from PIL import Image

class AdaIN:
    def __init__(self, config):
        self.config = config
        self.vgg = vgg
        self.head_vgg = nn.Sequential(*list(vgg.children())[:31])
        self.decoder = decoder
        self.style_image = None
        
    def loadModel(self, paths=None):
        if paths is None:
            self.vgg.load_state_dict(self.config.vgg)
            self.decoder.load_state_dict(self.config.decoder)
        else:
            self.vgg.load_state_dict(torch.load(paths["vgg"]))
            self.decoder.load_state_dict(torch.load(paths["decoder"]))
        self.head_vgg = nn.Sequential(*list(vgg.children())[:31])
    
    def _loadStyleImage(self):
        if self.style_image is not None:
            return self.style_image
        
        self.style_image = Image.open(self.config.style)
        self.style_image.convert("RGB")
        
        composite = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.config.style_size, self.config.style_size)),
        ])
        
        self.style_image = composite(self.style_image)
        return self.style_image
    
    def photographicToArtistic(self, content):
        return self.transformTo(content, self._loadStyleImage())
    
    def artisticToPhotographic(self, content):
        return self.transformTo(content, self._loadStyleImage())
    
    def transformTo(self, content, style):
        alpha = self.config.alpha
        device = self.config.device
        
        self.head_vgg.eval()
        self.decoder.eval()
        
        self.head_vgg.to(device)
        self.decoder.to(device)
        
        if self.config.preserve_color:
            style = coral(style, content)
        if content.ndim == 3:
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
        else:
            style = style.repeat((content.shape[0], 1, 1, 1)).to(device)
        with torch.no_grad():
            content_f = self.head_vgg(content)
            style_f = self.head_vgg(style)
            feat = adaptive_instance_normalization(content_f, style_f)
            feat = feat * alpha + content_f * (1 - alpha)
            return self.decoder(feat)