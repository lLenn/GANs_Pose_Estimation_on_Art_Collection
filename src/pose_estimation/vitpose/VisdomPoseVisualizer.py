from mmpose.visualization import PoseLocalVisualizer
from mmpose.registry import VISUALIZERS

@VISUALIZERS.register_module()
class VisdomPoseVisualizer(PoseLocalVisualizer):
    def push_images_to_backend(self):
        for vis_backend in self._vis_backends.values():
            if hasattr(vis_backend, "push"):
                vis_backend.push()
            
    def save_backend(self):
        for vis_backend in self._vis_backends.values():
            if hasattr(vis_backend, "save"):
                vis_backend.save()