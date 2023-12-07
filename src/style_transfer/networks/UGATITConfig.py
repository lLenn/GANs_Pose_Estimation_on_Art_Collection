from yacs.config import CfgNode as CN

class UGATITConfig:
    def create(file, options = list()):
        config = UGATITConfig.createDefaultConfig()
        config.defrost()
        config.merge_from_file(file)
        config.merge_from_list(options)
        config.freeze()
        return config
    
    def createDefaultConfig():
        default_config = CN()
        
        default_config.phase = "train"
        default_config.light = False
        default_config.dataset = 'YOUR_DATASET_NAME'

        default_config.iteration = 1000000
        default_config.batch_size = 1
        default_config.log_freq = 1
        default_config.print_freq = 1000
        default_config.save_freq = 100000
        default_config.decay_flag = True

        default_config.lr = 0.0001
        default_config.weight_decay = 0.0001
        default_config.adv_weight = 1
        default_config.cycle_weight = 10
        default_config.identity_weight = 10
        default_config.cam_weight = 1000
        
        default_config.ch = 64
        default_config.n_res = 4
        default_config.n_dis = 6

        default_config.img_size = 256
        default_config.img_ch = 3

        default_config.data_dir = "data"
        default_config.result_dir = "results"
        default_config.device = "cuda"
        default_config.benchmark_flag = False
        default_config.resume = False
        
        return default_config