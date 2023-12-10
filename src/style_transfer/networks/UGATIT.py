import os, cv2, time, torch
import numpy as np
from glob import glob
from collections import deque
from UGATITLib.UGATIT import UGATIT as UGATITNetwork
from UGATITLib.utils import *
from .UGATITVisualizer import UGATITVisualizer

def sortByEpochAndIteration(name):
    splitName = name.split(".")[-2].split("_")
    return (int(splitName[-2]), int(splitName[-1]))

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    cam_img = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
    return cam_img / 255.0

class UGATIT():
    def __init__(self, config):
        self.config = config
        self.model = UGATITNetwork(config)
        self.model.build_model()
        
        maxlen = 1
        if hasattr(config, "save_no"):
            maxlen = (None if config.save_no<0 else config.save_no)
        self.savedFiles = deque(maxlen = maxlen)
        
    def _load(self, epoch, iter):  
        self.loadModel(os.path.join(self.model.result_dir, self.model.dataset, 'model', f"{self.model.dataset}_params_{epoch}_{iter}.pt"))
        
    def _save(self, epoch, iter):  
        self.saveModel(os.path.join(self.model.result_dir, self.model.dataset, 'model', f"{self.model.dataset}_params_{epoch}_{iter}.pt"))
        
    def loadModel(self, path=None):
        if path is None:
            params = torch.load(self.config.model_path)
        else:
            params = torch.load(path)
        self.model.genA2B.load_state_dict(params['genA2B'])
        self.model.genB2A.load_state_dict(params['genB2A'])
        self.model.disGA.load_state_dict(params['disGA'])
        self.model.disGB.load_state_dict(params['disGB'])
        self.model.disLA.load_state_dict(params['disLA'])
        self.model.disLB.load_state_dict(params['disLB'])
        
    def saveModel(self, path):
        if len(self.savedFiles) == self.savedFiles.maxlen:
            os.remove(self.savedFiles.popleft())
        params = {}
        params['genA2B'] = self.model.genA2B.state_dict()
        params['genB2A'] = self.model.genB2A.state_dict()
        params['disGA'] = self.model.disGA.state_dict()
        params['disGB'] = self.model.disGB.state_dict()
        params['disLA'] = self.model.disLA.state_dict()
        params['disLB'] = self.model.disLB.state_dict()
        torch.save(params, path)
        self.savedFiles.append(path)
    
    def transformFromPhotographicToArtistic(self, image):
        self.model.genA2B.eval()
        image, _, _ = self.model.genA2B(image)
        return image
    
    def transformFromArtisticToPhotographic(self, image):
        self.model.genB2A.eval()
        image, _, _ = self.model.genB2A(image)
        return image
        
    def train(self, dataloader):
        visualizer = UGATITVisualizer(self.config)
        
        self.model.genA2B.train()
        self.model.genB2A.train()
        self.model.disGA.train()
        self.model.disGB.train()
        self.model.disLA.train()
        self.model.disLB.train()

        start_epoch = 1
        start_iter = 0
        if self.model.resume:
            model_list = glob(os.path.join(self.model.result_dir, self.model.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort(key=sortByEpochAndIteration)
                model_split = model_list[-1].split('.')[-2].split('_')
                start_epoch = int(model_split[-2])
                start_iter = int(model_split[-1])
                self._load(start_epoch, start_iter)
                print(f" [*] Load SUCCESS: epoch = {start_epoch}, iter = {start_iter}")
                start_iter += 1
                if self.model.decay_flag and start_epoch > (self.config.epoch // 2):
                    learning_rate = (self.model.lr / (self.config.epoch // 2)) * (start_epoch - self.config.epoch // 2)
                    self.model.G_optim.param_groups[0]['lr'] -= learning_rate
                    self.model.D_optim.param_groups[0]['lr'] -= learning_rate
                    print(f" Learning rate: G = {self.model.G_optim.param_groups[0]['lr']}, D = {self.model.D_optim.param_groups[0]['lr']}")

        # training loop
        print('training start !')
        start_time = time.time()
        dataset_size = len(dataloader)
        total_iter = (start_epoch-1)*dataset_size + start_iter
        for epoch in range(start_epoch, self.config.epoch + 1):
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            
            if self.model.decay_flag and epoch > (self.config.epoch // 2):
                self.model.G_optim.param_groups[0]['lr'] -= (self.model.lr / (self.config.epoch // 2))
                self.model.D_optim.param_groups[0]['lr'] -= (self.model.lr / (self.config.epoch // 2))
                print(f" Learning rate: G = {self.model.G_optim.param_groups[0]['lr']}, D = {self.model.D_optim.param_groups[0]['lr']}")

            epoch_iter = start_iter if epoch == start_epoch else 0
            dataloaderIterator = enumerate(dataloader)
            for _ in range(epoch_iter):
                next(dataloaderIterator)
                
            for iter, data in dataloaderIterator:
                total_iter += dataloader.batch_size
                epoch_iter += dataloader.batch_size
                
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iter % self.model.log_freq == 0:
                    t_data = iter_start_time - iter_data_time
            
                
                real_A = data["A"].to(self.model.device)
                real_B = data["B"].to(self.model.device)

                # Update D
                self.model.D_optim.zero_grad()

                fake_A2B, _, _ = self.model.genA2B(real_A)
                
                real_GB_logit, real_GB_cam_logit, _ = self.model.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.model.disLB(real_B)
                
                fake_GB_logit, fake_GB_cam_logit, _ = self.model.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.model.disLB(fake_A2B)
                
                D_ad_loss_GB = self.model.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.model.device)) + self.model.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.model.device))
                D_ad_cam_loss_GB = self.model.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.model.device)) + self.model.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.model.device))
                D_ad_loss_LB = self.model.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.model.device)) + self.model.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.model.device))
                D_ad_cam_loss_LB = self.model.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.model.device)) + self.model.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.model.device))
                
                D_loss_B = self.model.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)
                
                fake_B2A, _, _ = self.model.genB2A(real_B)

                real_GA_logit, real_GA_cam_logit, _ = self.model.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.model.disLA(real_A)

                fake_GA_logit, fake_GA_cam_logit, _ = self.model.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.model.disLA(fake_B2A)

                D_ad_loss_GA = self.model.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.model.device)) + self.model.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.model.device))
                D_ad_cam_loss_GA = self.model.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.model.device)) + self.model.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.model.device))
                D_ad_loss_LA = self.model.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.model.device)) + self.model.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.model.device))
                D_ad_cam_loss_LA = self.model.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.model.device)) + self.model.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.model.device))

                D_loss_A = self.model.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)

                Discriminator_loss = D_loss_A + D_loss_B
                Discriminator_loss.backward()
                self.model.D_optim.step()

                # Update G
                self.model.G_optim.zero_grad()

                fake_A2B, fake_A2B_cam_logit, _ = self.model.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.model.genB2A(real_B)

                fake_A2B2A, _, _ = self.model.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.model.genA2B(fake_B2A)

                fake_A2A, fake_A2A_cam_logit, _ = self.model.genB2A(real_A)
                fake_B2B, fake_B2B_cam_logit, _ = self.model.genA2B(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.model.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.model.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.model.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.model.disLB(fake_A2B)

                G_ad_loss_GA = self.model.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.model.device))
                G_ad_cam_loss_GA = self.model.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.model.device))
                G_ad_loss_LA = self.model.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.model.device))
                G_ad_cam_loss_LA = self.model.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.model.device))
                G_ad_loss_GB = self.model.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.model.device))
                G_ad_cam_loss_GB = self.model.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.model.device))
                G_ad_loss_LB = self.model.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.model.device))
                G_ad_cam_loss_LB = self.model.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.model.device))

                G_recon_loss_A = self.model.L1_loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.model.L1_loss(fake_B2A2B, real_B)

                G_identity_loss_A = self.model.L1_loss(fake_A2A, real_A)
                G_identity_loss_B = self.model.L1_loss(fake_B2B, real_B)

                G_cam_loss_A = self.model.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.model.device)) + self.model.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.model.device))
                G_cam_loss_B = self.model.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.model.device)) + self.model.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.model.device))

                G_loss_A =  self.model.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.model.cycle_weight * G_recon_loss_A + self.model.identity_weight * G_identity_loss_A + self.model.cam_weight * G_cam_loss_A
                G_loss_B = self.model.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.model.cycle_weight * G_recon_loss_B + self.model.identity_weight * G_identity_loss_B + self.model.cam_weight * G_cam_loss_B

                Generator_loss = G_loss_A + G_loss_B
                Generator_loss.backward()
                self.model.G_optim.step()

                # clip parameter of AdaILN and ILN, applied after optimizer step
                self.model.genA2B.apply(self.model.Rho_clipper)
                self.model.genB2A.apply(self.model.Rho_clipper)

                if total_iter % self.model.log_freq == 0:
                    losses = dict(
                        G_A_loss = float(G_loss_A),
                        G_B_loss = float(G_loss_B),
                        G_loss = float(Generator_loss),
                        D_A_loss = float(D_loss_A),
                        D_B_loss = float(D_loss_B),
                        D_loss = float(Discriminator_loss)
                    )
                    t_comp = (time.time() - iter_start_time) / dataloader.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                    
                if total_iter % self.model.print_freq == 0:
                    A2B = np.zeros((self.model.img_size * 7, 0, 3))
                    B2A = np.zeros((self.model.img_size * 7, 0, 3))

                    self.model.genA2B.eval(), self.model.genB2A.eval(), self.model.disGA.eval(), self.model.disGB.eval(), self.model.disLA.eval(), self.model.disLB.eval()

                    fake_A2B, _, fake_A2B_heatmap = self.model.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.model.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.model.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.model.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.model.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.model.genA2B(real_B)

                    visuals = dict(
                        realA = tensor2numpy(denorm(real_A[0])),
                        fakeA2AHeatmap = cam(tensor2numpy(fake_A2A_heatmap[0]), self.model.img_size),
                        fakeA2A = tensor2numpy(denorm(fake_A2A[0])),
                        fakeA2BHeatmap = cam(tensor2numpy(fake_A2B_heatmap[0]), self.model.img_size),
                        fakeA2B = tensor2numpy(denorm(fake_A2B[0])),
                        fakeA2B2AHeatmap = cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.model.img_size),
                        fakeA2B2A = tensor2numpy(denorm(fake_A2B2A[0])),
                        realB = tensor2numpy(denorm(real_B[0])),
                        fakeB2BHeatmap = cam(tensor2numpy(fake_B2B_heatmap[0]), self.model.img_size),
                        fakeB2B = tensor2numpy(denorm(fake_B2B[0])),
                        fakeB2AHeatmap = cam(tensor2numpy(fake_B2A_heatmap[0]), self.model.img_size),
                        fakeB2A = tensor2numpy(denorm(fake_B2A[0])),
                        fakeB2A2BHeatmap = cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.model.img_size),
                        fakeB2A2B = tensor2numpy(denorm(fake_B2A2B[0]))
                    )
                    visualizer.display_current_results(visuals)
                    self.model.genA2B.train(), self.model.genB2A.train(), self.model.disGA.train(), self.model.disGB.train(), self.model.disLA.train(), self.model.disLB.train()

                if total_iter % self.model.save_freq == 0:
                    self._save(epoch, iter)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, self.config.epoch, time.time() - epoch_start_time))
     
    def print(self):
        print("Unsupervised generative attentional network with adaptive layer-instance for image-to-image translation")
    
    def visualize(self, image, name):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.model.result_dir, f"{name}.png"), image)