import os, torch, time, datetime
import numpy as np
from glob import glob
from collections import deque
from UGATITLib.utils import denorm, tensor2numpy
from StarGAN.core.utils import denormalize
from StarGAN.core.solver import Solver, compute_d_loss, compute_g_loss, moving_average
from StarGAN.core.data_loader import InputFetcher
from StarGAN.metrics.eval import calculate_metrics
from .StarGANVisualizer import StarGANVisualizer

class StarGAN():
    def __init__(self, config):
        self.config = config
        self.model = Solver(config)
        if self.config.mode == "train":
            self.model_names = ["nets", "nets_ema", "optims"]
        else:
            self.model_names = ["nets_ema"]
            
        for name, module in self.model.nets.items():
            setattr(self.model, name, torch.nn.parallel.DistributedDataParallel(module.module))
        for name, module in self.model.nets_ema.items():
            setattr(self.model, name + '_ema', torch.nn.parallel.DistributedDataParallel(module.module))
            
        maxlen = 1
        if hasattr(config, "save_no"):
            maxlen = (None if config.save_no<0 else config.save_no)
        self.savedFiles = deque(maxlen = maxlen)
         
    def saveModel(self, suffix):
        if len(self.savedFiles) == self.savedFiles.maxlen:
            toRemove = self.savedFiles.popleft()
            for name in self.model_names:
                if isinstance(name, str):
                    os.remove(toRemove[name])
            
        toAdd = dict()
        for name in self.model_names:
            if isinstance(name, str):
                save_path = os.path.join(self.config.checkpoint_dir, f"{name}_{suffix}.ckpt")
                net = getattr(self.model, name)
                outdict = {}
                for item, module in net.items():
                    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
                        module = module.module
                    outdict[item] = module.state_dict()
                torch.save(outdict, save_path)
                toAdd[name] = save_path
        self.savedFiles.append(toAdd)
        
    def loadModel(self, paths=None, suffix=None):
        if paths is None:
            paths = self.config.checkpoint_dir
        if isinstance(paths, str) and suffix is None:
            suffix = self.config.resume_epoch
            
        toAdd = dict()
        for name in self.model_names:
            if isinstance(name, str):
                if suffix is None:
                    load_path = paths[name]
                else:
                    load_path = os.path.join(paths, f"{name}_{suffix}.ckpt")
                net = getattr(self.model, name)
                print('loading the model from %s...' % load_path)
                state_dict = torch.load(load_path, map_location=self.model.device)

                for item, module in net.items():
                    if isinstance(module, torch.nn.DataParallel):
                        module = module.module
                    module.load_state_dict(state_dict[item])
                toAdd[name] = load_path
        self.savedFiles.append(toAdd)
        
    def photographicToArtistic(self, image):
        return self.imageToStyle(image, torch.tensor([1]).to(self.model.device))
    
    def artisticToPhotographic(self, image):
        return self.imageToStyle(image, torch.tensor([0]).to(self.model.device))
    
    def imageToStyle(self, image, style):
        noise = torch.randn(1, self.config.latent_dim).to(self.model.device)
        styleEnc = self.model.mapping_network_ema(noise, style)
        styledImage = self.model.generator_ema(image, styleEnc)
        return styledImage
    
    def to(self, device):
        self.model.device = device
        self.model.to(device)
        for name, module in self.model.nets.items():
            module.to(device)
        for name, module in self.model.nets_ema.items():
            module.to(device)
    
    def train(self, dataloader_src, dataloader_ref, dataloader_val):
        args = self.model.args
        nets = self.model.nets
        nets_ema = self.model.nets_ema
        optims = self.model.optims
        
        start_iter = args.resume_iter
        if self.config.continue_training:
            model_list = glob(os.path.join(self.config.checkpoint_dir, '*.ckpt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('.')[-2].split('_')[-1])
                self.loadModel(self.config.checkpoint_dir, start_iter)
                print(f" [*] Load SUCCESS: iter = {start_iter}")
                
        visualizer = StarGANVisualizer(self.config)    # create a visualizer that display/save images and plots
        
        # fetch random validation images for debugging
        dataset_size = len(dataloader_src.dataset)
        fetcher = InputFetcher(dataloader_src, dataloader_ref, self.config.latent_dim, 'train')
        fetcher_val = InputFetcher(dataloader_val, None, self.config.latent_dim, 'val')
        
        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds
        lambda_ds = args.lambda_ds

        # training loop
        print('Start training...')
        start_time = time.time()            
        for step in range(start_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=z_trg)
            self.model._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(nets, args, x_real, y_org, y_trg, x_ref=x_ref)
            self.model._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2])
            self.model._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2])
            self.model._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if lambda_ds > 0:
                lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (step+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref], ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = lambda_ds
                visualizer.print_current_losses(step+1, all_losses, elapsed)
                visualizer.plot_current_losses(step+1, float(step+1) / dataset_size, all_losses)
            
            # generate images for debugging
            if (step+1) % args.sample_every == 0:
                nets_ema.generator.eval()
                nets_ema.style_encoder.eval()
                nets_ema.mapping_network.eval()
                
                inputs_val = next(fetcher_val)
                
                x_src, y_src = inputs_val.x_src, inputs_val.y_src
                x_ref, y_ref = inputs_val.x_ref, inputs_val.y_ref

                device = inputs_val.x_src.device
                N, C, H, W = inputs_val.x_src.size()

                s_ref = nets_ema.style_encoder(x_ref, y_ref)
                s_src = nets_ema.style_encoder(x_src, y_src)
                               
                x_fake = nets_ema.generator(x_src, s_ref)
                x_rec = nets_ema.generator(x_fake, s_src)
                
                visuals = dict(
                    x_src = tensor2numpy(denormalize(x_src[0])),
                    x_ref = tensor2numpy(denormalize(x_ref[0])),
                    x_fake = tensor2numpy(denormalize(x_fake[0])),
                    x_rec = tensor2numpy(denormalize(x_rec[0]))
                )
                visualizer.display_current_results("train", visuals, 4)
                
                # latent-guided image synthesis
                y_trg_list = [torch.tensor(y).repeat(N).to(device) for y in range(min(args.num_domains, 5))]
                z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
                visuals = dict()
                psi_vals = [0.5, 0.7, 1.0]
                # psi_vals = [1.0]
                for psi in psi_vals:
                    latent_dim = z_trg_list[0].size(1)
                    
                    for i, y_trg in enumerate(y_trg_list):
                        z_many = torch.randn(10000, latent_dim).to(device)
                        y_many = torch.LongTensor(10000).to(device).fill_(y_trg[0])
                        s_many = nets_ema.mapping_network(z_many, y_many)
                        s_avg = torch.mean(s_many, dim=0, keepdim=True)
                        s_avg = s_avg.repeat(N, 1)

                        for j, z_trg in enumerate(z_trg_list):
                            s_trg = nets_ema.mapping_network(z_trg, y_trg)
                            s_trg = torch.lerp(s_avg, s_trg, psi)
                            x_fake = nets_ema.generator(x_src, s_trg)
                            visuals[f"domain_{i+1}_val_{j+1}_latent_psi_{psi}"] = tensor2numpy(denormalize(x_fake[0]))

                visualizer.display_current_results("val", visuals, len(y_trg_list))
                
                # reference-guided image synthesis
                s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
                visuals = dict(
                    x_src = tensor2numpy(denormalize(x_src[0]))
                )
                for i, s_ref in enumerate(s_ref_list):
                    x_fake = nets_ema.generator(x_src, s_ref)
                    visuals[f"domain_{y_ref[0]}_{i+1}_reference"] = tensor2numpy(denormalize(x_fake[0]))
                visualizer.display_current_results("ref", visuals, len(s_ref_list)+1)
                
                nets_ema.generator.train()
                nets_ema.style_encoder.train()
                nets_ema.mapping_network.train()

            # save model checkpoints
            if (step+1) % args.save_every == 0:
                self.saveModel(step+1)
                visualizer.save()

            # compute FID and LPIPS if necessary
            if (step+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, step+1, mode='latent')
                calculate_metrics(nets_ema, args, step+1, mode='reference')
        
        