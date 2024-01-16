from mmengine.runner.log_processor import LogProcessor
from mmengine.registry import LOG_PROCESSORS

@LOG_PROCESSORS.register_module()
class VisdomLogProcessor(LogProcessor):
    def get_log_after_iter(self, runner, batch_idx, mode):
        tag, log_str = super().get_log_after_iter(runner, batch_idx, mode)
        tag["dataloader_len"] = self._get_dataloader_size(runner, mode)
        return tag, log_str