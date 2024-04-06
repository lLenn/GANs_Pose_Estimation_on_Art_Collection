import torch, time, os, cv2
from glob import glob
from torch.utils.data import DataLoader
from collections import deque
from CycleGAN.models import networks
from CycleGAN.models.cycle_gan_model import CycleGANModel
from .CycleGANVisualizer import CycleGANVisualizer

class CycleGAN:
    def __init__(self, config):
        self.config = config
        self.model = CycleGANModel(config)
        if self.config.isTrain:
            self.model.schedulers = [networks.get_scheduler(optimizer, config) for optimizer in self.model.optimizers]
        maxlen = 1
        if hasattr(config, "save_no"):
            maxlen = (None if config.save_no<0 else config.save_no)
        self.savedFiles = deque(maxlen = maxlen)
        
    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if len(self.savedFiles) == self.savedFiles.maxlen:
            toRemove = self.savedFiles.popleft()
            for name in self.model.model_names:
                if isinstance(name, str):
                    os.remove(toRemove[name])
            
        toAdd = dict()
        for name in self.model.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.model.save_dir, save_filename)
                net = getattr(self.model, 'net' + name)

                if len(self.model.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.model.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                toAdd[name] = save_path
        self.savedFiles.append(toAdd)
                     
    def loadModel(self, paths=None, suffix=None, withName=True):
        if paths is None:
            paths = self.config.models_dir
        if isinstance(paths, str) and suffix is None:
            suffix = self.config.epoch
        
        toAdd = dict()
        for name in self.model.model_names:
            if isinstance(name, str):
                if suffix is None:
                    load_path = paths[name]
                elif withName:
                    load_path = os.path.join(paths, self.config.name, f"{suffix}_net_{name}.pth")
                else:
                    load_path = os.path.join(paths, f"{suffix}_net_{name}.pth")
                net = getattr(self.model, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=(self.model.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.model.patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
                toAdd[name] = load_path
        self.savedFiles.append(toAdd)
    
    def photographicToArtistic(self, image):
        self.model.netG_B.eval()
        image = self.model.netG_B(image)
        return image
    
    def artisticToPhotographic(self, image):
        self.model.netG_A.eval()
        image = self.model.netG_A(image)
        return image
    
    def train(self, dataloader):
        dataset_size = len(dataloader.dataset)    # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)
    
        start_epoch = self.config.epoch_count
        if self.config.continue_train:
            model_list = glob(os.path.join(self.config.checkpoints_dir, self.config.name, '*.pth'))
            if not len(model_list) == 0:
                model_list.sort()
                load_epoch = int(os.path.split(model_list[-1])[1].split('_')[0])
                self.loadModel(self.config.checkpoints_dir, load_epoch)
                start_epoch = load_epoch+1
                
        visualizer = CycleGANVisualizer(self.config)    # create a visualizer that display/save images and plots
        total_iters = 0     # the total number of training iterations
        for epoch in range(start_epoch, self.config.n_epochs + self.config.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            self.model.update_learning_rate(epoch)    # update learning rates in the beginning of every epoch.
            for i, data in enumerate(dataloader):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % self.config.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += self.config.batch_size
                epoch_iter += self.config.batch_size
                self.model.set_input(data)         # unpack data from dataset and apply preprocessing
                self.model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                if total_iters % self.config.display_freq == 0:   # display images on visdom
                    print("results")
                    visualizer.display_current_results(self.model.get_current_visuals())

                if total_iters % self.config.print_freq == 0:    # print training losses and save logging information to the disk
                    print("loses")
                    losses = self.model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / self.config.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if self.config.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                    
                if total_iters % self.config.display_freq == 0 or total_iters % self.config.print_freq == 0:
                    print("save")

                iter_data_time = time.time()
                
            if epoch % self.config.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                self.save_networks(epoch)
                visualizer.save()

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, self.config.n_epochs + self.config.n_epochs_decay, time.time() - epoch_start_time))
    
    def visualize(self, image, name):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.config.results_dir, f"{name}.png"), image)
    