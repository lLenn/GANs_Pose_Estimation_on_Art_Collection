from yacs.config import CfgNode as CN

class StarGANConfig:
    def create(file, options = list()):
        config = StarGANConfig.createDefaultConfig()
        config.defrost()
        config.merge_from_file(file)
        config.merge_from_list(options)
        config.freeze()
        return config
    
    def createDefaultConfig():
        default_config = CN()
        
        default_config.device = "cuda"
        # model arguments
        default_config.img_size = 256 # Image resolution
        default_config.num_domains = 2  # Number of domains
        default_config.latent_dim = 16  # Latent vector dimension
        default_config.hidden_dim = 512  # Hidden dimension of mapping network
        default_config.style_dim = 64  # Style code dimension

        # weight for objective functions
        default_config.lambda_reg = 1  # Weight for R1 regularization
        default_config.lambda_cyc = 1  # Weight for cyclic consistency loss
        default_config.lambda_sty = 1  # Weight for style reconstruction loss
        default_config.lambda_ds = 1  # Weight for diversity sensitive loss
        default_config.ds_iter = 100000  # Number of iterations to optimize diversity sensitive loss
        default_config.w_hpf = 1  # weight for high-pass filtering

        # training arguments
        default_config.randcrop_prob = 0.5  # Probabilty of using random-resized cropping
        default_config.total_epoch = 2000  # Number of total iterations
        default_config.epoch = 0  # Iterations to resume training/testing
        default_config.batch_size = 8  # Batch size for training
        default_config.val_batch_size = 32  # Batch size for validation
        default_config.lr = 0.0001  # Learning rate for D, E and G
        default_config.f_lr = 0.000001  # Learning rate for F
        default_config.beta1 = 0.0  # Decay rate for 1st moment of Adam
        default_config.beta2 = 0.99  # Decay rate for 2nd moment of Adam
        default_config.weight_decay = 0.0001  # Weight decay for optimizer
        default_config.num_outs_per_domain = 10 # Number of generated images per domain during sampling

        # misc
        default_config.mode = 'train' # This argument is used in solver. Options: 'train', 'sample', 'eval', 'align'
        default_config.num_workers = 4  # Number of workers used in DataLoader
        default_config.seed = 777  # Seed for random number generator

        # directory for training
        default_config.train_img_dir = 'data/cel'  # Directory containing training images
        default_config.val_img_dir = 'data/celeb'  # Directory containing validation images
        default_config.sample_dir = '.output/sample'  # Directory for saving generated images
        default_config.checkpoint_dir = '.output/ch'  # Directory for saving network checkpoints

        # directory for calculating metrics
        default_config.eval_dir = '.output/eval'  # Directory for saving metrics, i.e., FID and LPIPS

        # directory for testing
        default_config.result_dir = '.output/result'  # Directory for saving generated images and videos
        default_config.src_dir = 'assets/represe'  # Directory containing input source images
        default_config.ref_dir = 'assets/represe'  # Directory containing input reference images
        default_config.inp_dir = 'assets/represe'  # input directory when aligning faces
        default_config.out_dir = 'assets/represe'  # output directory when aligning faces

        # step size
        default_config.save_no = 1
        default_config.print_every = 10
        default_config.sample_every = 5000
        default_config.save_every = 10000
        default_config.eval_every = 50000
    
        return default_config