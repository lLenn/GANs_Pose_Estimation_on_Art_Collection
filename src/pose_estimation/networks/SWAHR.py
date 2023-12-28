import torch, os, cv2, time
import numpy as np
import torchvision.transforms as transforms
from glob import glob
from utils import isArrayLike
from collections import deque
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from SWAHR.models.pose_higher_hrnet import PoseHigherResolutionNet
from SWAHR.core.group import HeatmapParser
from SWAHR.core.inference import get_multi_stage_outputs, aggregate_results
from SWAHR.core.loss import MultiLossFactory
from SWAHR.core.trainer import do_train
from SWAHR.utils.transforms import get_multi_scale_size, resize_align_multi_scale, get_final_preds
from SWAHR.utils.utils import get_model_summary, get_optimizer, AverageMeter
from .SWAHRVisualizer import SWAHRVisualizer

class SWAHR():
    def __init__(self, name, config):        
        self.name = name
        self.model = PoseHigherResolutionNet(config)
        self.config = config
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.heatmapParser = HeatmapParser(self.config)
        
        maxlen = (None if config.TRAIN.SAVE_NO<0 else config.TRAIN.SAVE_NO)
        self.savedFiles = deque(maxlen = maxlen)
    
    def saveModel(self, epoch):
        if len(self.savedFiles) == self.savedFiles.maxlen:
            os.remove(self.savedFiles.popleft())
                   
        file = os.path.join(self.config.TRAIN.CHECKPOINT, f"{epoch}_pose_higher_hrnet_{self.name}.pth") 
        torch.save(self.model.cpu().state_dict(), file)
        self.savedFiles.append(file)
        self.model.cuda()
        
    def loadModel(self, file = None, isTrain = False):
        if file == None:
            if isTrain:
                file = self.config.MODEL.PRETRAINED
            else:
                file = self.config.TEST.MODEL_FILE
        self.model.load_state_dict(torch.load(file), strict=True)
        self.savedFiles.append(file)
        
    def preprocess(self):
        pass
    
    def train(self, rank, world_size, dataloader, logger):
        dataset_size = len(dataloader.dataset)
        if self.config.VERBOSE:
            logger.info('The number of training images = %d' % dataset_size)
    
        start_epoch = self.config.TRAIN.BEGIN_EPOCH
        if self.config.TRAIN.RESUME:
            model_list = glob(os.path.join(self.config.TRAIN.CHECKPOINT, f"*_pose_higher_hrnet_{self.name}.pth"))
            if not len(model_list) == 0:
                model_list.sort()
                load_epoch = int(os.path.split(model_list[-1])[1].split('_')[0])
                self.loadModel(os.path.join(self.config.TRAIN.CHECKPOINT, f"{load_epoch}_pose_higher_hrnet_{self.name}.pth"))
                start_epoch = load_epoch
                 
        dump_input = torch.rand((1, 3, self.config.DATASET.INPUT_SIZE, self.config.DATASET.INPUT_SIZE))
        logger.info(get_model_summary(self.model, dump_input, verbose=self.config.VERBOSE))

        if rank == 0:
            visualizer = SWAHRVisualizer(self.config)

        # define loss function (criterion) and optimizer
        if world_size == 1:
            model = torch.nn.DataParallel(self.model).cuda(rank)
        else:
            self.model.cuda(rank)
            model = DistributedDataParallel(self.model, device_ids=[rank])
        loss_factory = MultiLossFactory(self.config).cuda()
        optimizer = get_optimizer(self.config, model)

        end_epoch = self.config.TRAIN.END_EPOCH
        warm_up_epoch = self.config.TRAIN.WARM_UP_EPOCH
        iters_per_epoch = len(dataloader)
        warm_up_iters = warm_up_epoch * iters_per_epoch
        train_iters = (end_epoch - warm_up_epoch) * iters_per_epoch
        initial_lr = self.config.TRAIN.LR
        
        for epoch in range(start_epoch, end_epoch):
            if world_size > 1:
                dataloader.sampler.set_epoch(epoch)
            
            batch_time = AverageMeter()
            data_time = AverageMeter()

            heatmaps_loss_meter = [AverageMeter() for _ in range(self.config.LOSS.NUM_STAGES)]
            scale_loss_meter = [AverageMeter() for _ in range(self.config.LOSS.NUM_STAGES)]
            push_loss_meter = [AverageMeter() for _ in range(self.config.LOSS.NUM_STAGES)]
            pull_loss_meter = [AverageMeter() for _ in range(self.config.LOSS.NUM_STAGES)]

            # switch to train mode
            model.train()

            end = time.time()
            for i, (images, heatmaps, masks, joints) in tqdm(enumerate(dataloader)):
                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                outputs = model(images)

                heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))
                masks = list(map(lambda x: x.cuda(non_blocking=True), masks))
                joints = list(map(lambda x: x.cuda(non_blocking=True), joints))

                # loss = loss_factory(outputs, heatmaps, masks)
                heatmaps_losses, scale_losses, push_losses, pull_losses = loss_factory(outputs, heatmaps, masks, joints)

                loss = 0
                for idx in range(self.config.LOSS.NUM_STAGES):
                    if heatmaps_losses[idx] is not None:
                        heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                        scale_loss = scale_losses[idx].mean(dim=0)

                        heatmaps_loss_meter[idx].update(
                            heatmaps_loss.item(), images.size(0)
                        )
                        scale_loss_meter[idx].update(
                            scale_loss.item(), images.size(0)
                        )

                        loss = loss + heatmaps_loss + scale_loss
                        if push_losses[idx] is not None:
                            push_loss = push_losses[idx].mean(dim=0)
                            push_loss_meter[idx].update(
                                push_loss.item(), images.size(0)
                            )
                            loss = loss + push_loss
                        if pull_losses[idx] is not None:
                            pull_loss = pull_losses[idx].mean(dim=0)
                            pull_loss_meter[idx].update(
                                pull_loss.item(), images.size(0)
                            )
                            loss = loss + pull_loss

                # compute gradient and do update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # adjust learning rate
                current_iters = epoch * iters_per_epoch + i
                if current_iters < warm_up_iters:
                    lr = initial_lr * 0.1 + current_iters / warm_up_iters * initial_lr * 0.9
                else:
                    lr = (1 - (current_iters - warm_up_iters) / train_iters) * initial_lr
                    
                for param in optimizer.param_groups:
                    param['lr'] = lr

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if rank == 0 and i % self.config.PRINT_FREQ == 0:
                    losses = {
                        "heatmaps": heatmaps_loss_meter,
                        "scale": scale_loss_meter,
                        "push": push_loss_meter,
                        "pull": pull_loss_meter
                    }
                    visualizer.print_current_losses(epoch, i, iters_per_epoch, lr, batch_time, images.size(0)/batch_time.val, data_time, losses)

                    for key in losses:
                        loss = {}
                        for idx in range(self.config.LOSS.NUM_STAGES):
                            loss[f"stage{idx}-{key}"] = heatmaps_loss_meter[idx].val
                        visualizer.plot_current_losses(f"{key} loss over time", key, epoch, i/iters_per_epoch, loss)
                    
                    for scale_idx in range(len(outputs)):
                        prefix_scale = f"train_output_{self.config.DATASET.OUTPUT_SIZE[scale_idx]}"
                        num_joints = self.config.DATASET.NUM_JOINTS
                        batch_pred_heatmaps = outputs[scale_idx][:, :num_joints, :, :]
                        batch_pred_tagmaps = outputs[scale_idx][:, num_joints:, :, :]

                        if self.config.DEBUG.SAVE_HEATMAPS_GT and heatmaps[scale_idx] is not None:
                            visualizer.display_current_results(f'{prefix_scale}_hm_gt.jpg', visualizer.save_batch_maps(images, heatmaps[scale_idx], masks[scale_idx], 'heatmap'), 1)
                        if self.config.DEBUG.SAVE_HEATMAPS_PRED:
                            visualizer.display_current_results(f'{prefix_scale}_hm_pred.jpg', visualizer.save_batch_maps(images, batch_pred_heatmaps, masks[scale_idx], 'heatmap'), 1)
                        if self.config.DEBUG.SAVE_TAGMAPS_PRED:
                            visualizer.display_current_results(f'{prefix_scale}_tag_pred.jpg', visualizer.save_batch_maps(images, batch_pred_tagmaps, masks[scale_idx], 'tagmap'), 1)

                    visualizer.save()
                    
            if rank == 0 and epoch % self.config.SAVE_FREQ == 0:
                self.saveModel(epoch+1)
    
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
        dataloader = torch.utils.data.DataLoader(
            sub_dataset, sampler=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
        )
        predictions = []
        pbar = tqdm(total=len(sub_dataset)) if self.config.TEST.LOG_PROGRESS else None
        for i, (images, annotations) in enumerate(dataloader):
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