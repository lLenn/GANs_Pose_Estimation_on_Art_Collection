import cv2
from torchvision.transforms import transforms
from PIL import Image
from SWAHR.dataset.COCODataset import CocoDataset
from SWAHR.utils import zipreader

class COCOMetaDataset(CocoDataset):
    def __init__(self, root, dataset, data_format):
        super().__init__(root, dataset, data_format)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        
        image_info = coco.loadImgs(img_id)[0]
        file_name = image_info['file_name']
        
        if self.data_format == 'zip':
            img = zipreader.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            img = cv2.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            '''
            img = Image.open(self._get_image_path(file_name))
            img.convert("RGB")
            img = transforms.ToTensor()(img)
            '''

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img, [image_info]

    def __len__(self):
        return len(self.ids)