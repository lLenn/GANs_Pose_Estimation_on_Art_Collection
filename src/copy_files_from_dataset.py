import os
import shutil

# script for presentation
# copies images from different styles to dir for easier comparison

unstyled_images = [776, 15497, 26690, 35682, 40757, 45550, 108026, 148662, 158956, 162092, 177015, 133343]
artifact_images = [32570, 37988, 38210, 140556, 177489]
over_stylized_images = [23899, 31093, 32081, 32735, 65736, 100428, 149770, 172877]
stylized_images = [14226, 16598, 27972, 29397, 31217, 60347, 93261, 109313, 115898, 127270, 146155, 153529, 161875, 170099, 170613, 174371, 67180, 122046, 173799, 181421, 190307, 199055, 142620]

COPY_TO = "../../Experiments/Style Transfer"
COPY_FROM = "../../Datasets/coco/val_adain"
ID_ADDITION = 8000000000000
ID_SUB_ADDITION = 100000000000

for cat in range(4):
    images = []
    dir = []
    if cat == 0:
        images = unstyled_images
        dir = "Unchanged"
    elif cat == 1:
        images = artifact_images
        dir = "Artifacts"
    elif cat == 2:
        images = over_stylized_images
        dir = "Overstylized"
    elif cat == 3:
        images = stylized_images
        dir = "Styled"
    for image in images:
        for style in range(3):
            style_image = image + ID_ADDITION + ID_SUB_ADDITION*style
            shutil.copy(os.path.join(COPY_FROM, f"{style_image}.jpg"), os.path.join(COPY_TO, dir))