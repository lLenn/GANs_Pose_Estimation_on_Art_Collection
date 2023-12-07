import numpy as np
import os
import time
from CycleGAN.util import util

class CycleGANVisualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display
    """

    def __init__(self, config):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.config = config  # cache the option
        self.display_id = config.display_id
        self.win_size = config.display_winsize
        self.name = config.name
        self.port = config.display_port
        self.ncols = config.display_ncols

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.vis = visdom.Visdom(server=config.display_server, port=config.display_port, env=config.display_env)
            if not self.vis.check_connection():
                raise ConnectionError(f"Can't connect to Visdom server at {config.display_server}:{config.display_port}")

        # create a logging file to store training losses
        self.log_name = os.path.join(config.checkpoints_dir, config.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, visuals):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
        """
        if self.display_id == 0:
            return
        
        ncols = self.ncols
        if ncols > 0:        # show all the images in one visdom panel
            ncols = min(ncols, len(visuals))
            h, w = next(iter(visuals.values())).shape[:2]
            table_css = """<style>
                    table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                    table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                    </style>""" % (w, h)  # create a table css
            # create a table of images.
            title = self.name
            label_html = ''
            label_html_row = ''
            images = []
            idx = 0
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                label_html_row += '<td>%s</td>' % label
                images.append(image_numpy.transpose([2, 0, 1]))
                idx += 1
                if idx % ncols == 0:
                    label_html += '<tr>%s</tr>' % label_html_row
                    label_html_row = ''
            white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
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

        else:     # show each image in a separate visdom panel;
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                win=f"{self.name}_image_{label}")

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if self.display_id == 0:
            return
        
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
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
            
    def save(self):
        self.vis.save([self.config.display_env])
