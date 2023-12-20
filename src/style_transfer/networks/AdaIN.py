import torch
import torch.nn as nn
from AdaIN.net import vgg, decoder
from AdaIN.function import adaptive_instance_normalization, coral

class AdaIN:
    def __init__(self, config):
        self.config = config
        self.vgg = vgg
        self.decoder = decoder
        
    def loadModel(self, paths=None):
        if paths is None:
            self.vgg.load_state_dict(self.config.vgg)
            self.decoder.load_state_dict(self.config.decoder)
        else:
            self.vgg.load_state_dict(torch.load(paths["vgg"]))
            self.decoder.load_state_dict(torch.load(paths["decoder"]))
            
    def transformTo(self, content, style):
        alpha = self.config.alpha
        print(alpha)
        device = self.config.device
        
        vgg = nn.Sequential(*list(self.vgg.children())[:31])
        vgg.eval()
        self.decoder.eval()
        
        vgg.to(device)
        self.decoder.to(device)
        
        if self.config.preserve_color:
            style = coral(style, content)
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            content_f = vgg(content)
            style_f = vgg(style)
            feat = adaptive_instance_normalization(content_f, style_f)
            feat = feat * alpha + content_f * (1 - alpha)
            return self.decoder(feat)