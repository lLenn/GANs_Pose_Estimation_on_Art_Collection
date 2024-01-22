import os
from SWAHR.dataset.COCODataset import CocoDataset

class ArtPoseDataset(CocoDataset):
    def __init__(self, root, dataset, file):
        self.file = file
        super().__init__(root, dataset, "jpg")
            
    def _get_anno_file_name(self):
        return os.path.join(self.root, self.file)
    
    def _get_image_path(self, file_name):
        if self.data_format == 'zip':
            return os.path.join(self.root, 'images') + '.zip@' + file_name
        else:
            return os.path.join(self.root, 'images', file_name)