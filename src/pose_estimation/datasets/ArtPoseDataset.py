import os
from SWAHR.dataset.COCODataset import CocoDataset

class ArtPoseDataset(CocoDataset):
    def __init__(self, root, dataset, file, data_format, transform=None, target_transform=None):
        self.file = file
        super().__init__(root, dataset, "jpg")
            
    def _get_anno_file_name(self):
        return os.path.join(self.root, "annotations", self.file)