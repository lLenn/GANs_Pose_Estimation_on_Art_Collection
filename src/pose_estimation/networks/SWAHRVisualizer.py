import visdom, os, cv2
import numpy as np
from datetime import datetime
from SWAHR.utils.vis import make_heatmaps, make_tagmaps

class SWAHRVisualizer():
    def __init__(self, config):
        self.config = config  # cache the option
        self.name = config.VISDOM.NAME
        self.port = config.VISDOM.PORT

        self.vis = visdom.Visdom(server=config.VISDOM.SERVER, port=config.VISDOM.PORT, env=config.VISDOM.ENV)
        if not self.vis.check_connection():
            raise ConnectionError(f"Can't connect to Visdom server at {config.VISDOM.SERVER}:{config.VISDOM.PORT}")
        
        # create a logging file to store training losses
        self.log_name = os.path.join(config.LOG_DIR, f"swahr_loss_log_{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')}.txt")

    def display_current_results(self, title, visuals, n_cols):
        # create a table of images.
        images = []
        for image in visuals:
            images.append(image.transpose([2, 0, 1]))
        self.vis.images(images, nrow=n_cols, win=f"{self.name}_{title}_images", padding=2, opts=dict(title=f"{self.name}: {title} images"))

    def plot_current_losses(self, title, name, epoch, counter_ratio, losses):
        losKeys = list(losses.keys())
        no_losses = len(losKeys)
        
        self.vis.line(
            X=np.array([[epoch + counter_ratio] * no_losses]),
            Y=np.array([[losses[k] for k in losKeys]]),
            opts={
                'title': title,
                'legend': losKeys,
                'xlabel': 'epoch',
                'ylabel': 'loss'
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

    def save_batch_maps(self, batch_image, batch_maps, batch_mask, map_type='heatmap', normalize=True):
        if normalize:
            batch_image = batch_image.clone()
            min_val = float(batch_image.min())
            max_val = float(batch_image.max())
            batch_image.add_(-min_val).div_(max_val - min_val + 1e-5)

        batch_size = batch_maps.size(0)
        num_joints = batch_maps.size(1)
        map_height = batch_maps.size(2)
        map_width = batch_maps.size(3)
        
        grid = np.empty((0, map_height, (num_joints+1)*map_width, 3))
        for i in range(min(3, batch_size)):
            image = batch_image[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            maps = batch_maps[i]

            if map_type == 'heatmap':
                image_with_hms = make_heatmaps(image, maps)
            elif map_type == 'tagmap':
                image_with_hms = make_tagmaps(image, maps)

            if batch_mask is not None:
                mask = np.expand_dims(batch_mask[i].byte().cpu().numpy(), -1)
                image_with_hms[:, :map_width, :] = image_with_hms[:, :map_width, :] * mask
                
            image_with_hms = cv2.cvtColor(image_with_hms, cv2.COLOR_BGR2RGB)
            grid = np.append(grid, [image_with_hms], axis=0)
            
        return grid       
        
    def save(self):
        self.vis.save([self.config.VISDOM.ENV])
