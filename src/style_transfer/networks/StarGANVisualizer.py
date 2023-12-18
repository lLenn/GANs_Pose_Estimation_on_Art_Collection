import visdom, time, os
import numpy as np
from CycleGAN.util import util

class StarGANVisualizer():
    def __init__(self, config):
        self.config = config  # cache the option
        self.name = config.name
        self.port = config.display_port

        self.vis = visdom.Visdom(server=config.display_server, port=config.display_port, env=config.display_env)
        if not self.vis.check_connection():
            raise ConnectionError(f"Can't connect to Visdom server at {config.display_server}:{config.display_port}")
        
        # create a logging file to store training losses
        self.log_name = os.path.join(config.checkpoint_dir, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, title, visuals, n_cols):
        table_css = """<style>
                table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                table td {padding: 4px; outline: 4px solid black}
                </style>"""  # create a table css
        # create a table of images.
        label_html = ''
        label_html_row = ''
        images = []
        idx = 0
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            label_html_row += '<td>%s</td>' % label
            images.append(image_numpy.transpose([2, 0, 1]))
            idx += 1
            if idx % n_cols == 0:
                label_html += '<tr>%s</tr>' % label_html_row
                label_html_row = ''
        white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
        while idx % n_cols != 0:
            images.append(white_image)
            label_html_row += '<td></td>'
            idx += 1
        if label_html_row != '':
            label_html += '<tr>%s</tr>' % label_html_row
        self.vis.images(images, nrow=n_cols, win=f"{self.name}_{title}_images", padding=2, opts=dict(title=f"{self.name}: {title} images"))
        label_html = '<table>%s</table>' % label_html
        self.vis.text(table_css + label_html, win=f"{self.name}_{title}_labels", opts=dict(title=f"{self.name}: {title} labels"))

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """        
        losKeys = list(losses.keys())
        no_losses = len(losKeys)
        
        self.vis.line(
            X=np.array([[epoch + counter_ratio] * no_losses]),
            Y=np.array([[losses[k] for k in losKeys]]),
            opts={
                'title': self.name + ' loss over time',
                'legend': losKeys,
                'xlabel': 'epoch',
                'ylabel': 'loss'
            },
            win=f"{self.name}_loss",
            update='append'
        )

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, iters, losses, t_comp):
        """print current losses on console; also save the losses to the disk

        Parameters:
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
        """
        message = '(iters: %d, time: %.3f) ' % (iters, t_comp)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
        
    def save(self):
        self.vis.save([self.config.display_env])
