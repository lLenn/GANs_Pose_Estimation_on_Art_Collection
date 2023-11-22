from pose_estimation.networks import SWAHR, SWAHRConfig
from style_transfer.datasets import HumanArtDataset
from style_transfer.networks import UGATIT, UGATITConfig

AVAILABLE_DATASETS = [
    [HumanArtDataset, "../../datasets/human-art", "oil_painting", "test", False]
]

AVAILABLE_POSE_ESTIMATORS = [
    # [SWAHR, SWAHRConfig, "src/pose_estimation/config/w32_512_test.yaml", list()],
    [SWAHR, SWAHRConfig, "src/pose_estimation/config/w48_640_test.yaml", list()]
]

AVAILABLE_STYLE_TRANSFORMERS = [
    [UGATIT, UGATITConfig, "src/style_transfer/config/ugatit.yaml", list()]
]

def createDatasetIterator():
    def iterator():
        for datasetConfig in AVAILABLE_DATASETS:
            yield datasetConfig[0](*datasetConfig[1:])
    return iterator()

def createPoseEstimatorIterator():
    def iterator():
        for poseEstimatorConfig in AVAILABLE_POSE_ESTIMATORS:
            yield poseEstimatorConfig[0](poseEstimatorConfig[1].create(poseEstimatorConfig[2], poseEstimatorConfig[3]))
    return iterator()

def createStyleTransformerIterator():
    def iterator():
        for styleTransformerConfig in AVAILABLE_STYLE_TRANSFORMERS:
            yield styleTransformerConfig[0](styleTransformerConfig[1].create(styleTransformerConfig[2], styleTransformerConfig[3]))
    return iterator()
