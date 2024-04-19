import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

class ImageDirectoryDataset(Dataset):
    def __init__(self, root, file_ext = ".jpg", size=512):
        self.root = root
        self.files = [file for file in os.listdir(self.root) if file.endswith(file_ext)]
        self.size = size
    
    def __getitem__(self, index):
        file_name = self.files[index]
        file_path = os.path.join(self.root, file_name)
        
        image = Image.open(file_path)
        image.convert("RGB")
        
        composite = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.size, self.size)),
        ])
        
        return composite(image), { "filename": file_name }
          
    def __len__(self):
        return len(self.files)