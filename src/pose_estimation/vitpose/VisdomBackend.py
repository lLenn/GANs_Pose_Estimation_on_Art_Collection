import visdom, torch
import numpy as np
from typing import Optional, Sequence, Union
from mmengine.config import Config
from mmengine.registry import VISBACKENDS
from mmengine.visualization.vis_backend import BaseVisBackend, force_init_env

@VISBACKENDS.register_module()
class VisdomBackend(BaseVisBackend):
    def __init__(self,
                 save_dir: str,
                 init_kwargs: Optional[dict] = None):
        super().__init__(save_dir)
        self.image_idx = dict()
        self.scalars = dict()
        self._init_kwargs = init_kwargs

    def _init_env(self):
        self.vis = visdom.Visdom(server=self._init_kwargs.server, port=self._init_kwargs.port, env=self._init_kwargs.env)
        if not self.vis.check_connection():  
            raise ConnectionError(f"Can't connect to Visdom server at {self._init_kwargs.server}:{self._init_kwargs.port}")

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        return self.vis

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        pass
    
    @force_init_env
    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict], **kwargs) -> None:
        pass

    @force_init_env
    def add_image(self, name: str, image: np.ndarray, step: int = 0, **kwargs) -> None:
        if name not in self.image_idx:
            self.image_idx[name] = 0
        self.vis.images(image.transpose([2, 0, 1]), nrow=1, win=f"{self._init_kwargs.name}_{name}_image_{self.image_idx[name]}", padding=2, opts=dict(title=f"{self._init_kwargs.name} {name} image {self.image_idx[name]}"))
        self.image_idx[name] += 1

    @force_init_env
    def add_scalar(self, name: str, value: Union[int, float, torch.Tensor, np.ndarray], step: int = 0, **kwargs) -> None:
        pass

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        
        scalarKeys = list(scalar_dict.keys())
        
        if "loss" in scalarKeys:
            self.vis.line(
                X=np.array([[scalar_dict["epoch"] + scalar_dict["batch_idx"]/scalar_dict["dataloader_len"]]]),
                Y=np.array([[scalar_dict["loss"]]]),
                opts={
                    'title': "Loss ViTPose",
                    'legend': ["loss"],
                    'xlabel': 'epoch',
                    'ylabel': 'loss',
                    'ytype': 'log'
                },
                win=f"{self._init_kwargs.name}_loss",
                update='append'
            )
        elif "coco/AP" in scalarKeys:
            scalarKeys = ["coco/AP", "coco/AP .5", "coco/AP .75", "coco/AP (M)", "coco/AP (L)", "coco/AR", "coco/AR .5", "coco/AR .75", "coco/AR (M)", "coco/AR (L)"]
            self.vis.line(
                X=np.array([[step]*len(scalarKeys)]),
                Y=np.array([[scalar_dict[k] for k in scalarKeys]]),
                opts={
                    'title': "Evaluation ViTPose",
                    'legend': scalarKeys,
                    'xlabel': 'epoch',
                    'ylabel': '%'
                },
                win=f"{self._init_kwargs.name}_eval",
                update='append'
            )

    @force_init_env
    def push(self) -> None:
        self.image_idx = dict()
     
    @force_init_env   
    def save(self):
        self.vis.save([self._init_kwargs.env])