from yacs.config import CfgNode as CN

class AdaINConfig:
    def create(file, options = list()):
        config = AdaINConfig.createDefaultConfig()
        config.defrost()
        config.merge_from_file(file)
        config.merge_from_list(options)
        config.freeze()
        return config
    
    def createDefaultConfig():
        default_config = CN()
        
        # basic options
        default_config.device = "cuda"
        default_config.content_dir = "" # Directory path to a batch of content images
        default_config.style = "" # "File path to the style image, or multiple style images separated by commas if you want to do style interpolation or spatial control
        default_config.style_dir = "" # Directory path to a batch of style images
        default_config.vgg = "models/vgg_normalised.pth"
        default_config.decoder = "models/decoder.pth"

        # Additional options
        default_config.content_size = 512 # New (minimum) size for the content image keeping the original size if set to 0
        default_config.style_size = 512 # New (minimum) size for the style image, keeping the original size if set to 0
        default_config.crop = True # do center crop to create squared image
        default_config.save_ext = ".jpg" # The extension name of the output image
        default_config.output = "output" # Directory to save the output image(s)

        # Advanced options
        default_config.preserve_color = True # If specified, preserve color of the content image
        default_config.alpha = 1.0 # The weight that controls the degree of stylization. Should be between 0 and 1
        default_config.style_interpolation_weights = "" # The weight for blending the style of multiple style images

        # training options
        default_config.save_dir = "./experiments" # Directory to save the model
        default_config.log_dir = "./logs" # Directory to save the log
        default_config.lr = 1e-4
        default_config.lr_decay = 5e-5
        default_config.max_iter = 160000
        default_config.batch_size = 8
        default_config.style_weight = 10.0
        default_config.content_weight = 1.0
        default_config.n_threads = 16
        default_config.save_model_interval = 10000
        
        # visdom
        default_config.display_server = "http://localhost"
        default_config.display_env = "main"
        default_config.display_port = 8087
    
        return default_config