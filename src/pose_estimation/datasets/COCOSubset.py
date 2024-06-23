import os
from SWAHR.dataset.COCODataset import CocoDataset

# Split the set into keypoint and normal
class COCOSubset(CocoDataset):
    META_INFO = {
        "keypoints": [
            ["nose", [51, 153, 255]],
            ["left_eye", [51, 153, 255]],
            ["right_eye", [51, 153, 255]],
            ["left_ear", [51, 153, 255]],
            ["right_ear", [51, 153, 255]],
            ["left_shoulder", [0, 255, 0]],
            ["right_shoulder", [255, 128, 0]],
            ["left_elbow", [0, 255, 0]],
            ["right_elbow", [255, 128, 0]],
            ["left_wrist", [0, 255, 0]],
            ["right_wrist", [255, 128, 0]],
            ["left_hip", [0, 255, 0]],
            ["right_hip", [255, 128, 0]],
            ["left_knee", [0, 255, 0]],
            ["right_knee", [255, 128, 0]],
            ["left_ankle", [0, 255, 0]],
            ["right_ankle", [255, 128, 0]]
        ],
        "skeleton": [
            [ 16, 14, [0, 255, 0] ],
            [ 14, 12, [0, 255, 0] ],
            [ 17, 15, [255, 128, 0] ],
            [ 15, 13, [255, 128, 0] ],
            [ 12, 13, [51, 153, 255] ],
            [ 6, 12, [51, 153, 255] ],
            [ 7, 13, [51, 153, 255] ],
            [ 6, 7, [51, 153, 255] ],
            [ 6, 8, [0, 255, 0] ],
            [ 7, 9, [255, 128, 0] ],
            [ 8, 10, [0, 255, 0] ],
            [ 9, 11, [255, 128, 0] ],
            [ 2, 3, [51, 153, 255] ],
            [ 1, 2, [51, 153, 255] ],
            [ 1, 3, [51, 153, 255] ],
            [ 2, 4, [51, 153, 255] ],
            [ 3, 5, [51, 153, 255] ],
            [ 4, 6, [51, 153, 255] ],
            [ 5, 7, [51, 153, 255] ]
        ],
        "joint_weights": [
            1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
            1.5
        ],
        "sigmas": [
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
            0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]
    }
    
    def _get_anno_file_name(self):
        return os.path.join(
            self.root,
            "annotations",
            f"{self.dataset}.json"
        )
        
    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images')
        return os.path.join(images_dir, 'val2017', file_name)