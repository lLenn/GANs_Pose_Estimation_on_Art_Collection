import torch
import os
import torchvision.transforms
import cv2
import numpy as np
from utils import isArrayLike
from tqdm import tqdm
from SWAHR.models.pose_higher_hrnet import PoseHigherResolutionNet
from SWAHR.core.group import HeatmapParser
from SWAHR.core.inference import get_multi_stage_outputs, aggregate_results
from SWAHR.utils.transforms import get_multi_scale_size, resize_align_multi_scale, get_final_preds

class SWAHR():
    def __init__(self, config):        
        self.model = PoseHigherResolutionNet(config)
        self.config = config
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.heatmapParser = HeatmapParser(self.config)
    
    def loadModel(self, isTrain):
        if isTrain:
            file = self.config.MODEL.PRETRAINED
        else:
            file = self.config.TEST.MODEL_FILE            
        self.model.load_state_dict(torch.load(file), strict=True)
    
    def preprocess(self):
        pass
    
    def train(self):
        pass
    
    def infer(self, gpuIds, image):
        gpuIds if isArrayLike(gpuIds) else [gpuIds]
        
        model = torch.nn.DataParallel(self.model, device_ids=gpuIds)
        model = model.cuda()
        
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, self.config.DATASET.INPUT_SIZE, 1.0, min(self.config.TEST.SCALE_FACTOR)
        )
        
        model.eval()
        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for idx, s in enumerate(sorted(self.config.TEST.SCALE_FACTOR, reverse=True)):
                input_size = self.config.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(self.config.TEST.SCALE_FACTOR)
                )
                image_resized = self.transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    self.config, model, image_resized, self.config.TEST.FLIP_TEST,
                    self.config.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    self.config, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(self.config.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)

            grouped, scores = self.heatmapParser.parse(
                final_heatmaps, tags, self.config.TEST.ADJUST, self.config.TEST.REFINE
            )

            final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )
            
            return image_resized, final_heatmaps, final_results, scores
    
    def validate(self, gpuIds, dataset, indices, logger):
        sub_dataset = torch.utils.data.Subset(dataset, indices)
        data_loader = torch.utils.data.DataLoader(
            sub_dataset, sampler=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
        )
        predictions = []
        pbar = tqdm(total=len(sub_dataset)) if self.config.TEST.LOG_PROGRESS else None
        for i, (images, annotations) in enumerate(data_loader):
            image = images[0].cpu().numpy()

            image_resized, final_heatmaps, final_results, scores = self.infer(gpuIds, image)

            visual = True
            if visual:
                visual_heatmap = torch.max(final_heatmaps[0], dim=0, keepdim=True)[0]
                visual_heatmap = (
                    visual_heatmap.cpu().numpy().repeat(3, 0).transpose(1, 2, 0)
                )

                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                visual_img = (
                    image_resized[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
                )
                visual_img = visual_img[:, :, ::-1] * np.array(std).reshape(
                    1, 1, 3
                ) + np.array(mean).reshape(1, 1, 3)
                visual_img = visual_img * 255
                test_data = cv2.addWeighted(
                    visual_img.astype(np.float32),
                    0.0,
                    visual_heatmap.astype(np.float32) * 255,
                    1.0,
                    0,
                )
                cv2.imwrite(os.path.join(self.config.OUTPUT_DIR, f"test_data/{int(annotations[0]['image_id'])}.jpg"), test_data)

            if self.config.TEST.LOG_PROGRESS:
                pbar.update()

            for idx in range(len(final_results)):
                predictions.append({
                    "keypoints": final_results[idx][:,:3].reshape(-1,).astype(float).tolist(),
                    "image_id": int(annotations[0]["image_id"]),
                    "score": float(scores[idx]),
                    "category_id": 1
                })

        if self.config.TEST.LOG_PROGRESS:
            pbar.close()
            
        return predictions

    def inference(self):
        pass
    
    def visualize(self, image, heatmaps, filename):
        visual_heatmap = torch.max(heatmaps, dim=0, keepdim=True)[0]
        visual_heatmap = (
            visual_heatmap.cpu().numpy().repeat(3, 0).transpose(1, 2, 0)
        )

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        visual_img = (
            image.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        )
        '''
        visual_img = visual_img[:, :, ::-1] * np.array(std).reshape(
            1, 1, 3
        ) + np.array(mean).reshape(1, 1, 3)
        '''
        visual_img = visual_img * 255
        test_data = cv2.addWeighted(
            visual_img.astype(np.float32),
            0.0,
            visual_heatmap.astype(np.float32) * 255,
            1.0,
            0,
        )
        cv2.imwrite(os.path.join(self.config.OUTPUT_DIR, f"test_data/{filename}.jpg"), test_data)
    
    def toString(self):
        return f"SWAHR w{self.config.MODEL.EXTRA.DECONV.NUM_CHANNELS} {self.config.DATASET.INPUT_SIZE}"