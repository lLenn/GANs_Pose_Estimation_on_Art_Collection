import numpy as np
import visdom
from CycleGAN.util import util

class UGATITVisualizer():
    def __init__(self, config):
        self.config = config  # cache the option
        self.name = config.name
        self.port = config.display_port

        self.vis = visdom.Visdom(server=config.display_server, port=config.display_port, env=config.display_env)
        if not self.vis.check_connection():
            raise ConnectionError(f"Can't connect to Visdom server at {config.display_server}:{config.display_port}")

    def display_current_results(self, visuals):
        ncols = 7
        table_css = """<style>
                table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                table td {padding: 4px; outline: 4px solid black}
                </style>"""  # create a table css
        # create a table of images.
        title = self.name
        label_html = ''
        label_html_row = ''
        images = []
        idx = 0
        for label, image in visuals.items():
            label_html_row += '<td>%s</td>' % label
            images.append(image.transpose([2, 0, 1]))
            idx += 1
            if idx % ncols == 0:
                label_html += '<tr>%s</tr>' % label_html_row
                label_html_row = ''
        white_image = np.ones_like(image.transpose([2, 0, 1])) * 255
        while idx % ncols != 0:
            images.append(white_image)
            label_html_row += '<td></td>'
            idx += 1
        if label_html_row != '':
            label_html += '<tr>%s</tr>' % label_html_row
        self.vis.images(images, nrow=ncols, win=f"{self.name}_images",
                        padding=2, opts=dict(title=title + ' images'))
        label_html = '<table>%s</table>' % label_html
        self.vis.text(table_css + label_html, win=f"{self.name}_labels",
                        opts=dict(title=title + ' labels'))

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
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        
    def save(self):
        self.vis.save([self.config.display_env])
