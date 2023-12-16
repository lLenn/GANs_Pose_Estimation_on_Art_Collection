import os, torch
from collections import deque
from StarGAN.core import Solver

class StarGAN():
    def __init__(self, config):
        self.config = config
        self.model = Solver(config)
        if self.config.mode == "train":
            self.model_names = ["nets", "nets_ema", "optims"]
        else:
            self.model_names = ["nets"]
        maxlen = 1
        if hasattr(config, "save_no"):
            maxlen = (None if config.save_no<0 else config.save_no)
        self.savedFiles = deque(maxlen = maxlen)
        
    def loadModel(self, paths=None, suffix=None):
        if paths is None:
            paths = self.config.checkpoint_dir
        if isinstance(paths, str) and suffix is None:
            suffix = self.config.epoch
            
        toAdd = dict()
        for name in self.model_names:
            if isinstance(name, str):
                if suffix is None:
                    load_path = paths[name]
                else:
                    load_path = os.path.join(paths, self.config.name, f"{suffix}_{name}.ckpt")
                net = getattr(self.model, name)
                print('loading the model from %s...' % load_path)
                state_dict = torch.load(load_path, map_location=self.model.device)

                for item, module in net.items():
                    if isinstance(module, torch.nn.DataParallel):
                        module = module.module
                    module.load_state_dict(state_dict[item])
                toAdd[name] = load_path
        self.savedFiles.append(toAdd)
        
    def photographicToArtistic(self, imageFrom, imageTo):
        noise = torch.randn(1, self.config.latent_dim).to(self.config.device)
        style = self.model.nets_ema.mapping_network(imageFrom, imageTo)
        