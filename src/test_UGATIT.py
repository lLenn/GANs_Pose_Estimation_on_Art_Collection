import cv2
import os
from PIL import Image
from style_transfer.networks import UGATIT, UGATITConfig
from torchvision.transforms import transforms
import torch

config = UGATITConfig.create("src/style_transfer/config/ugatit.yaml")
network = UGATIT(config)
# network.train()

network.loadModel("src/style_transfer/model")
with open("../../Datasets/human-art/images/2D_virtual_human/oil_painting/000000000000.jpg", 'rb') as file:
    image = Image.open(file)
    image.convert("RGB")
    
imagecv2 = cv2.imread("../../Datasets/human-art/images/2D_virtual_human/oil_painting/000000000000.jpg", cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
imagecv2 = cv2.cvtColor(imagecv2, cv2.COLOR_BGR2RGB)
    
image = transforms.ToTensor()(image)
image = transforms.Resize((256, 256))(image)
image = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)
image = torch.stack((image,))
image = image.to("cuda")

image = network.artisticToPhotographic(image)
image = image[0] * 0.5 + 0.5
image = image.detach().cpu().numpy()
image = image.transpose(1,2,0)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(config.result_dir, "test_ugatit.png"), image * 255.0)
