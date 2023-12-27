from yacs.config import CfgNode as CN
import torch.backends.cudnn as cudnn
from SWAHR.config import update_config, check_config

class SWAHRConfig:
    def create(file=None, options=list()):
        args = type('', (object, ), { "cfg": file, "opts": options })
        config = SWAHRConfig.createDefaultConfig()
        update_config(config, args)
        check_config(config)
        return config
    
    def configureEnvironment(config):
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED
    
    def createDefaultConfig():
        default_config = CN()

        default_config.OUTPUT_DIR = ''
        default_config.LOG_DIR = ''
        default_config.DATA_DIR = ''
        default_config.GPUS = (0,)
        default_config.WORKERS = 4
        default_config.WORLD_SIZE = 8
        default_config.PRINT_FREQ = 20
        default_config.SAVE_FREQ = 1
        default_config.AUTO_RESUME = False
        default_config.PIN_MEMORY = True
        default_config.RANK = 0
        default_config.VERBOSE = True
        default_config.DIST_BACKEND = 'nccl'
        default_config.MULTIPROCESSING_DISTRIBUTED = True

        # FP16 training params
        default_config.FP16 = CN()
        default_config.FP16.ENABLED = False
        default_config.FP16.STATIC_LOSS_SCALE = 1.0
        default_config.FP16.DYNAMIC_LOSS_SCALE = False

        # Cudnn related params
        default_config.CUDNN = CN()
        default_config.CUDNN.BENCHMARK = True
        default_config.CUDNN.DETERMINISTIC = False
        default_config.CUDNN.ENABLED = True

        # common params for NETWORK
        default_config.MODEL = CN()
        default_config.MODEL.NAME = 'pose_multi_resolution_net_v16'
        default_config.MODEL.INIT_WEIGHTS = True
        default_config.MODEL.PRETRAINED = ''
        default_config.MODEL.NUM_JOINTS = 17
        default_config.MODEL.TAG_PER_JOINT = True
        default_config.MODEL.EXTRA = CN(new_allowed=True)
        default_config.MODEL.SYNC_BN = False

        default_config.LOSS = CN()
        default_config.LOSS.NUM_STAGES = 1
        default_config.LOSS.WITH_HEATMAPS_LOSS = (True,)
        default_config.LOSS.HEATMAPS_LOSS_FACTOR = (1.0,)
        default_config.LOSS.WITH_AE_LOSS = (True,)
        default_config.LOSS.AE_LOSS_TYPE = 'max'
        default_config.LOSS.PUSH_LOSS_FACTOR = (0.001,)
        default_config.LOSS.PULL_LOSS_FACTOR = (0.001,)

        # DATASET related params
        default_config.DATASET = CN()
        default_config.DATASET.ROOT = ''
        default_config.DATASET.DATASET = 'coco_kpt'
        default_config.DATASET.DATASET_TEST = 'coco'
        default_config.DATASET.NUM_JOINTS = 17
        default_config.DATASET.MAX_NUM_PEOPLE = 30
        default_config.DATASET.TRAIN = 'train2017'
        default_config.DATASET.TEST = 'val2017'
        default_config.DATASET.DATA_FORMAT = 'jpg'

        # training data augmentation
        default_config.DATASET.MAX_ROTATION = 30
        default_config.DATASET.MIN_SCALE = 0.75
        default_config.DATASET.MAX_SCALE = 1.25
        default_config.DATASET.SCALE_TYPE = 'short'
        default_config.DATASET.MAX_TRANSLATE = 40
        default_config.DATASET.INPUT_SIZE = 512
        default_config.DATASET.OUTPUT_SIZE = [128, 256, 512]
        default_config.DATASET.FLIP = 0.5

        # heatmap generator (default is OUTPUT_SIZE/64)
        default_config.DATASET.SIGMA = -1
        default_config.DATASET.SCALE_AWARE_SIGMA = False
        default_config.DATASET.BASE_SIZE = 256.0
        default_config.DATASET.BASE_SIGMA = 2.0
        default_config.DATASET.INT_SIGMA = False

        default_config.DATASET.WITH_CENTER = False

        # train
        default_config.TRAIN = CN()

        default_config.TRAIN.LR_FACTOR = 0.1
        default_config.TRAIN.LR_STEP = [90, 110]
        default_config.TRAIN.LR = 0.001

        default_config.TRAIN.OPTIMIZER = 'adam'
        default_config.TRAIN.MOMENTUM = 0.9
        default_config.TRAIN.WD = 0.0001
        default_config.TRAIN.NESTEROV = False
        default_config.TRAIN.GAMMA1 = 0.99
        default_config.TRAIN.GAMMA2 = 0.0

        default_config.TRAIN.BEGIN_EPOCH = 0
        default_config.TRAIN.END_EPOCH = 140
        default_config.TRAIN.WARM_UP_EPOCH = 3

        default_config.TRAIN.RESUME = False
        default_config.TRAIN.SAVE_NO = 1
        default_config.TRAIN.CHECKPOINT = ''

        default_config.TRAIN.IMAGES_PER_GPU = 32
        default_config.TRAIN.SHUFFLE = True

        # testing
        default_config.TEST = CN()

        # size of images for each device
        # default_config.TEST.BATCH_SIZE = 32
        default_config.TEST.IMAGES_PER_GPU = 32
        # Test Model Epoch
        default_config.TEST.FLIP_TEST = False
        default_config.TEST.ADJUST = True
        default_config.TEST.REFINE = True
        default_config.TEST.SCALE_FACTOR = [1]
        # group
        default_config.TEST.DETECTION_THRESHOLD = 0.2
        default_config.TEST.TAG_THRESHOLD = 1.
        default_config.TEST.USE_DETECTION_VAL = True
        default_config.TEST.IGNORE_TOO_MUCH = False
        default_config.TEST.MODEL_FILE = ''
        default_config.TEST.IGNORE_CENTER = True
        default_config.TEST.NMS_KERNEL = 3
        default_config.TEST.NMS_PADDING = 1
        default_config.TEST.PROJECT2IMAGE = False

        default_config.TEST.WITH_HEATMAPS = (True,)
        default_config.TEST.WITH_AE = (True,)

        default_config.TEST.LOG_PROGRESS = True

        # debug
        default_config.DEBUG = CN()
        default_config.DEBUG.DEBUG = True
        default_config.DEBUG.SAVE_BATCH_IMAGES_GT = False
        default_config.DEBUG.SAVE_BATCH_IMAGES_PRED = False
        default_config.DEBUG.SAVE_HEATMAPS_GT = True
        default_config.DEBUG.SAVE_HEATMAPS_PRED = True
        default_config.DEBUG.SAVE_TAGMAPS_PRED = True
        
        default_config.VISDOM = CN()
        default_config.VISDOM.NAME = "experiments"
        default_config.VISDOM.SERVER = "http://localhost"
        default_config.VISDOM.PORT = "8097"
        default_config.VISDOM.ENV = "main"
        
        return default_config