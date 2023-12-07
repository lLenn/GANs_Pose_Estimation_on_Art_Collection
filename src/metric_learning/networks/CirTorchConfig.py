from yacs.config import CfgNode as CN

class CirTorchConfig:
    def create(file, options = list()):
        config = CirTorchConfig.createDefaultConfig()
        config.defrost()
        config.merge_from_file(file)
        config.merge_from_list(options)
        config.freeze()
        return config
        
    def createDefaultConfig():
        default_config = CN()

        default_config.network_path = ""
        default_config.image_size = 1024
        default_config.multiscale = [1]
        return default_config