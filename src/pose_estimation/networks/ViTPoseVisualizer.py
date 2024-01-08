import visdom, os, cv2
import numpy as np
from datetime import datetime
from mmpose.visualization import PoseLocalVisualizer

class ViTPoseVisualizer():
    def __init__(self, log_dir, config):
        self.log_dir = log_dir
        self.name = config.name
        self.port = config.port
        self.server = config.server
        self.env = config.env

        self.pose_local_visualizer = PoseLocalVisualizer()
        self.connected = False
        
        # create a logging file to store training losses
        self.log_name = os.path.join(self.log_dir, f"vitpose_loss_log_{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')}.txt")

    def connect(self):
        self.vis = visdom.Visdom(server=self.server, port=self.port, env=self.env)
        if not self.vis.check_connection():
            raise ConnectionError(f"Can't connect to Visdom server at {self.server}:{self.port}")

    def _check_connection(self):
        if not self.connected:
            raise ConnectionError(f"Not connected")

    def display_current_results(self, title, visuals, n_cols):
        self._check_connection()
        # create a table of images.
        images = []
        for image in visuals:
            images.append(image.transpose([2, 0, 1]))
        self.vis.images(images, nrow=n_cols, win=f"{self.name}_{title}_images", padding=2, opts=dict(title=f"{self.name}: {title} images"))

    def plot_current_losses(self, title, name, epoch, counter_ratio, losses):
        self._check_connection()
        losKeys = list(losses.keys())
        no_losses = len(losKeys)
        
        self.vis.line(
            X=np.array([[epoch + counter_ratio] * no_losses]),
            Y=np.array([[losses[k] for k in losKeys]]),
            opts={
                'title': title,
                'legend': losKeys,
                'xlabel': 'epoch',
                'ylabel': 'loss',
                'ytype': 'log'
            },
            win=f"{name}_loss",
            update='append'
        )

    def print_current_losses(self, epoch, iter, total_iter, lr, batch_time, speed, data_time, losses):
        message = f"Epoch: [{epoch}][{iter}/{total_iter}]\tlr:{lr:.4f}\tTime: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\tSpeed: {speed:.1f} samples/s\tData: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
        for key in losses:
            message += self._get_loss_info(losses[key], key)
            
        print(message) 
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
            
    def _get_loss_info(self, loss_meters, loss_name):
        msg = ''
        for i, meter in enumerate(loss_meters):
            msg += 'Stage{i}-{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
                i=i, name=loss_name, meter=meter
            )

        return msg

    # From adapted from ViTPose: https://github.com/open-mmlab/mmpose/blob/509441ed7c9087cd05fb3f9e22644fe0da9d8dd8/mmpose/visualization/local_visualizer.py
    def draw_predictions(self, image, predictions, draw_bbox=False):
        prediction_image = self.pose_local_visualizer._draw_instances_kpts(image, predictions, 0.3, False, "mmpose")
        if draw_bbox:            
            prediction_image = self._draw_instances_bbox(image, predictions)        
        return prediction_image
        
    def save(self):
        self._check_connection()
        self.vis.save([self.env])
