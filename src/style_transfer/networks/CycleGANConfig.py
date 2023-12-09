import math
from yacs.config import CfgNode as CN

class CycleGANConfig:
    def create(file, options = list(), phase="train", model="cycle_gan"):
        config = CycleGANConfig.createDefaultConfig(phase, model)
        config.defrost()
        config.merge_from_file(file)
        config.merge_from_list(options)
        config.freeze()
        return config
    
    def createDefaultConfig(phase="train", model="cycle_gan"):
        default_config = CN()
        
        default_config.dataroot = "data"
        default_config.name = "experiment_name"
        default_config.gpu_ids = [0]
        default_config.checkpoints_dir = "checkpoints"
        default_config.models_dir = "checkpoints"

        default_config.model = "cycle_gan"
        default_config.input_nc = 3
        default_config.output_nc = 3
        default_config.ngf = 64
        default_config.ndf = 64
        default_config.netD = "basic"
        default_config.netG = "resnet_9blocks"
        default_config.n_layers_D = 3
        default_config.norm = "instance"
        default_config.init_type = "normal"
        default_config.init_gain = 0.02
        default_config.no_dropout = True
        
        default_config.dataset_mode = "unaligned"
        default_config.direction = "AtoB"
        default_config.serial_batches = True
        default_config.num_threads = 4
        default_config.batch_size = 1
        default_config.load_size = 286
        default_config.crop_size = 256
        default_config.max_dataset_size = float("inf")
        default_config.preprocess = "resize_and_crop"
        default_config.no_flip = True
        default_config.display_winsize = 256

        default_config.epoch = "latest"
        default_config.verbose = True
        default_config.suffix = ""
        
        if phase == "train":
            default_config.display_freq = 400
            default_config.display_ncols = 4
            default_config.display_id = 1
            default_config.display_server = "http://localhost"
            default_config.display_env = "main"
            default_config.display_port = 8087
            default_config.print_freq = 100
            
            default_config.save_no = -1
            default_config.save_epoch_freq = 5
            default_config.continue_train = True
            default_config.epoch_count = 1
            default_config.phase = "train"
            
            default_config.n_epochs = 100
            default_config.n_epochs_decay = 100
            default_config.beta1 = 0.5
            default_config.lr = 0.0002
            default_config.gan_mode = "lsgan"
            default_config.pool_size = 50
            default_config.lr_policy = "linear"
            default_config.lr_decay_iters = 50
            
            default_config.isTrain = True
            
            if model == "cycle_gan":
                default_config.lambda_A = 10.0
                default_config.lambda_B = 10.0
                default_config.lambda_identity = 0.5
                
                
        elif phase == "test":
            default_config.results_dir = "./output/results"
            default_config.aspect_ratio = 1.0
            default_config.phase = "test"
            
            default_config.eval = True
            default_config.num_test = 50
        
            default_config.model = "test"
            default_config.load_size = default_config.crop_size
            
            default_config.isTrain = False
            
        return default_config