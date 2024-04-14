import os
from SWAHR.dataset.COCODataset import CocoDataset

# Split the set into keypoint and normal
class COCOSubset(CocoDataset):
    META_INFO = {
        "keypoints": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"
        ],
        "skeleton": [
            [ 16, 14 ],
            [ 14, 12 ],
            [ 17, 15 ],
            [ 15, 13 ],
            [ 12, 13 ],
            [ 6, 12 ],
            [ 7, 13 ],
            [ 6, 7 ],
            [ 6, 8 ],
            [ 7, 9 ],
            [ 8, 10 ],
            [ 9, 11 ],
            [ 2, 3 ],
            [ 1, 2 ],
            [ 1, 3 ],
            [ 2, 4 ],
            [ 3, 5 ],
            [ 4, 6 ],
            [ 5, 7 ]
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