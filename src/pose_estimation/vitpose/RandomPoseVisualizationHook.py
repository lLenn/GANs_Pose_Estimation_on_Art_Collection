# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
import random
from typing import Optional, Sequence

import mmcv
import mmengine
import mmengine.fileio as fileio
from mmengine.dist import get_dist_info, broadcast_object_list
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmpose.registry import HOOKS
from mmpose.structures import PoseDataSample, merge_data_samples


@HOOKS.register_module()
class RandomPoseVisualizationHook(Hook):
    """Random Pose Estimation Visualization Hook. Used to visualize random
    samples from validation and testing process prediction results.

    In the testing phase:

    1. If ``out_dir`` is specified, it means that the prediction results
        need to be saved to ``out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    2. ``vis_backends`` takes effect if the user does not specify ``show``
        and `out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        kpt_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the prefix of
            uri corresponding backend. Defaults to None.
    """

    def __init__(
        self,
        no_samples: int = 4,
        kpt_thr: float = 0.3,
        out_dir: Optional[str] = None,
        backend_args: Optional[dict] = None,
    ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.no_samples = no_samples
        self.kpt_thr = kpt_thr
        self.out_dir = out_dir
        self._test_index = 0
        self.backend_args = backend_args
        self.samples = []
        self.sampleIndices = []

    def before_val(self, runner: Runner) -> None:
        self._visualizer.set_dataset_meta(runner.val_evaluator.dataset_meta)
        
        rank, world_size = get_dist_info()
        self.sampleIndices = []
        
        if rank == 0:
            for _ in range(self.no_samples):
                self.sampleIndices.append(random.randint(0, len(runner.val_dataloader.dataset)-1))
            
            if world_size == 1:
                return
            
        else:
            self.sampleIndices = [None] * self.no_samples
            
        broadcast_object_list(self.sampleIndices, 0)
        
    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict, outputs: Sequence[PoseDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        batch_size = len(data_batch["inputs"])
        lower_val_iter = batch_idx*batch_size
        total_curr_iter = runner.iter + batch_idx

        for idx in range(lower_val_iter, (batch_idx+1)*batch_size):
            if idx not in self.sampleIndices:
                continue
            
            # Visualize only the first data
            img_path = data_batch['data_samples'][idx-lower_val_iter].get('img_path')
            img_bytes = fileio.get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            data_sample = outputs[idx-lower_val_iter]

            # revert the heatmap on the original image
            data_sample = merge_data_samples([data_sample])
            
            self._visualizer.add_datasample(
                "val_img",
                img,
                data_sample=data_sample,
                draw_gt=False,
                draw_bbox=True,
                draw_heatmap=True,
                show=False,
                wait_time=0,
                kpt_thr=self.kpt_thr,
                step=total_curr_iter)
            
    def after_val(self, runner):
        self._visualizer.push_images_to_backend()