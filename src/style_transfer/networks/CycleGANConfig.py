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
        default_config.gpu_ids = 0
        default_config.checkpoints_dir = "checkpoints"

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
        default_config.no_dropout = "store_true"
        
        default_config.dataset_mode = "unaligned"
        default_config.direction = "AtoB"
        default_config.serial_batches = "store_true"
        default_config.num_threads = 4
        default_config.batch_size = 1
        default_config.load_size = 286
        default_config.crop_size = 256
        default_config.max_dataset_size = float("inf")
        default_config.preprocess = "resize_and_crop"
        default_config.no_flip = "store_true"
        default_config.display_winsize = 256

        default_config.epoch = "latest"
        default_config.load_iter = 0
        default_config.verbose = "store_true"
        default_config.suffix = ""

        default_config.use_wandb = "store_true"
        default_config.wandb_project_name = "CycleGAN-and-pix2pix"
        
        if phase == "train":
            default_config.display_freq = 400
            default_config.display_ncols = 4
            default_config.display_id = 1
            default_config.display_server = "http://localhost"
            default_config.display_env = "main"
            default_config.display_port = 8087
            default_config.update_html_freq = 1000
            default_config.print_freq = 100
            default_config.no_html = "store_true"
            
            default_config.save_latest_freq = 5000
            default_config.save_epoch_freq = 5
            default_config.save_by_iter = "store_true"
            default_config.continue_train = "store_true"
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
            
            if model == "cycle_gan":
                default_config.lambda_A = 10.0
                default_config.lambda_B = 10.0
                default_config.lambda_identity = 0.5
                
                
        elif phase == "test":
            default_config.results_dir = "results"
            default_config.aspect_ratio = 1.0
            default_config.phase = "test"
            
            default_config.eval = "store_true"
            default_config.num_test = 50
        
            default_config.model = "test"
            default_config.load_size = default_config.crop_size
            
        return default_config