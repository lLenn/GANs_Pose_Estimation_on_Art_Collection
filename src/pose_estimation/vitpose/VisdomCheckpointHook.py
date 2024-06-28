
from mmengine.registry import HOOKS
from mmengine.hooks import CheckpointHook
from mmengine.visualization import Visualizer

# mmpose classes so results of training can be shared with a visdom server

@HOOKS.register_module()
class VisdomCheckpointHook(CheckpointHook):    
    def after_train_epoch(self, runner):
        super().after_train_epoch(runner)
        Visualizer.get_current_instance().save_backend()