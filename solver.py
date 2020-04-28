from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import time
import datetime
from scipy import io
       
class Solver(object):
    """Solver for training and testing Photometric Stereo."""

    def __init__(self, pixels_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.pixels_loader = pixels_loader


        # Model configurations.
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num  
        self.lambda_L1 = config.lambda_L1
        self.lambda_Light = config.lambda_Light
        self.lambda_Iso = config.lambda_Iso
        self.lambda_Sparse = config.lambda_Sparse
        self.lambda_Conti = config.lambda_Conti

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.result_ind = config.result_ind
        self.shape = config.shape

        # Step size.
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size) 
        self.m = nn.AvgPool2d(2,stride=2,padding=0)
            
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
  
    def angular_deviation(self, iput, target, dataset='Pixels'):
        """Compute Regression loss."""
        if dataset == 'Pixels':
            iput_norm = torch.norm(iput,dim=1)
            iput_norm = iput_norm.unsqueeze(1)
            iput_norm = iput_norm.expand(iput_norm.size(0),iput.size(1))
            target_norm = torch.norm(target,dim=1)
            target_norm = target_norm.unsqueeze(1)
            target_norm = target_norm.expand(target_norm.size(0),target.size(1))
            cosdelta = iput*target/(iput_norm*target_norm)
            cosdelta = torch.sum(cosdelta,dim=1)
            for ii in range(cosdelta.size(0)):
                if cosdelta[ii].data<=1 and cosdelta[ii].data>=-1:
                    cosdelta[ii] = cosdelta[ii]
                else:
                    cosdelta[ii] = torch.tensor(-1.).to(self.device)
            delta = torch.acos(cosdelta)
            return delta/np.pi*180
            
    def regression_loss(self, iput, target, dataset='Pixels'):
        """Compute Regression loss."""
        if dataset == 'Pixels':
            loss_fn = torch.nn.CosineEmbeddingLoss()
            y = torch.ones(iput.size(0)).to(self.device)
            cosdelta =  1-loss_fn(iput, target, y)
            delta = torch.acos(cosdelta)
            return delta/np.pi*180
      
    def L1_loss(self, iput, target, dataset = 'Pixels'):
        """Compute L1 loss."""
        if dataset == 'Pixels':
            loss_fn = torch.nn.L1Loss(reduction='elementwise_mean')
            return loss_fn(iput, target)

    def Light_loss(self, iput, target, x, y, dataset = 'Pixels'):
        """Compute Light loss."""
        if dataset == 'Pixels':
            s = torch.zeros(1).to(self.device)
            for k in range(iput.size(0)):
                s = s + torch.sum(torch.abs(iput[k,0,x[k],y[k]]-target[k,0,x[k],y[k]]))
            return s/(iput.size(0)*10)
            
    def Iso_loss(self, iput, target, N, dataset = 'Pixels'):
        """Compute Iso loss."""
        if dataset == 'Pixels':
            s1 = torch.zeros(1).to(self.device)
            s2 = torch.zeros(1).to(self.device)
            s3 = torch.zeros(1).to(self.device)
            iput2 = self.m(iput)
            target2 = self.m(target) 
            for k in range(iput.size(0)):
                n = N[k]
                ind = torch.nonzero(target[k].squeeze(0)).to(self.device)
                ind2 = torch.nonzero(target2[k].squeeze(0)).to(self.device)
                try:
                    x = ind[:,0]
                    y = ind[:,1]
                    ind_ = (ind2.float().to(self.device)/15*2-1)
                    ind_norm = torch.norm(ind_,dim=1)
                    mask = torch.lt(ind_norm,1)
                    ind2 = ind2[mask]
                    x2 = ind2[:,0]
                    y2 = ind2[:,1]
                except:
                    x = ((n[1]+1)*0.5/15).long().to(self.device)
                    y = ((n[0]+1)*0.5/15).long().to(self.device)
                    x2 = ((n[1]+1)*0.5/15).long().to(self.device)
                    y2 = ((n[0]+1)*0.5/15).long().to(self.device)
                xs, ys = self.Get_Iso_Coordinate(x, y, n)
                xs2, ys2 = self.Get_Iso_Coordinate(x2, y2, n)
                s1 = s1 + torch.sum(torch.pow(iput[k,0,x,y]-iput[k,0,xs,ys],2))
                s2 = s2 + torch.pow(torch.sum(torch.abs(iput[k,0,x,y]-iput[k,0,xs,ys]))-50,2)
                s3 = s3 + torch.pow(torch.sum(torch.abs(iput2[k,0,x2,y2]-iput[k,0,xs2,ys2]))-10,2)
        return s1/(iput.size(0)), s2/(iput.size(0)), s3/(iput.size(0))
    
    def Get_Iso_Coordinate(self, x, y, n):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x = (x.float().to(self.device)/15*2-1)
        y = (y.float().to(self.device)/15*2-1)             
        axis = n[0:2]
        axis = axis/torch.norm(axis)
        theta = torch.acos(axis[0]).to(self.device)
        if axis[1]<0:
            theta = -theta + 2*np.pi
        rotm1 = torch.tensor([[torch.cos(theta),-torch.sin(theta)],[torch.sin(theta),torch.cos(theta)]]).to(self.device)
        temp1 = torch.matmul(rotm1, torch.cat([y,x],dim=0)).to(self.device)
        try:
            xt = temp1[1,:]
            yt = temp1[0,:]
        except:
            xt = temp1[1]
            yt = temp1[0]
        xt = -xt
        rotm2 = torch.tensor([[torch.cos(-theta),-torch.sin(-theta)],[torch.sin(-theta),torch.cos(-theta)]]).to(self.device)
        temp2 = torch.matmul(rotm2, torch.cat([yt.unsqueeze(0),xt.unsqueeze(0)],dim=0)).to(self.device)
        try:
            xs = (temp2[1,:]+1)*0.5/15
            ys = (temp2[0,:]+1)*0.5/15
        except:
            xs = (temp2[1]+1)*0.5/15
            ys = (temp2[0]+1)*0.5/15
        xs = xs.long().to(self.device)
        ys = ys.long().to(self.device)
        return xs, ys
            

    def train(self):  
        """Train Photometric Stereo within a single dataset."""
        # Set data loader.
        data_loader = self.pixels_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        xall_fixed, x10_fixed, n_org, X_fixed, Y_fixed= next(data_iter)
        xall_fixed = xall_fixed.to(self.device)
        x10_fixed = x10_fixed.to(self.device)
        n_org = n_org.to(self.device)
        X_fixed = X_fixed.to(self.device)
        Y_fixed = Y_fixed.to(self.device)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
             
             # Fetch images and labels
            try:
                xall_real, x10_real, normal_org, X, Y= next(data_iter)
            except:
                data_iter = iter(data_loader)
                xall_real, x10_real, normal_org, X, Y = next(data_iter)
            
            xall_real = xall_real.to(self.device)     # Input alllight images.
            x10_real = x10_real.to(self.device)       # Input 10light images.
            normal_org = normal_org.to(self.device)   # Original normals.
            X = X.to(self.device)                     # 10lights' coordinates  
            Y = Y.to(self.device)



            # =================================================================================== #
            #                             2. Train the Normal Estimation Net                      #
            # =================================================================================== #

            # Compute loss with fake images.
            xall_fake= self.G(x10_real)
            xdin_fake=torch.cat([xall_fake,x10_real],dim=1)
            out_reg = self.D(xdin_fake.detach())
            d_loss_fake = self.regression_loss(out_reg, normal_org, self.dataset)

            
            #Isotropy loss
            d_loss_Iso, d_loss_Sparse, d_loss_Conti = self.Iso_loss(xall_real, xall_real, out_reg, self.dataset)
            
            # Backward and optimize.
            d_loss = d_loss_fake + self.lambda_Iso * d_loss_Iso\
                    + self.lambda_Sparse * d_loss_Sparse + self.lambda_Conti * d_loss_Conti
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/fake'] = d_loss_fake.item()
            loss['D/Iso'] = d_loss_Iso.item()
            loss['D/Sparse'] = d_loss_Sparse.item()
            loss['D/Conti'] = d_loss_Conti.item()
            
            # =================================================================================== #
            #                               3. Train the Lighting Interpolation Net               #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                xall_fake = self.G(x10_real)
                xdin_fake=torch.cat([xall_fake,x10_real],dim=1)
                out_reg = self.D(xdin_fake)

                g_loss_fake = self.regression_loss(out_reg, normal_org, self.dataset)
                
                # L1 loss.
                g_loss_L1 = self.L1_loss(xall_fake, xall_real, self.dataset)
#
                # Light loss
                g_loss_Light = self.Light_loss(xall_fake, xall_real, X, Y, self.dataset)
                
                # Isotropy loss
                g_loss_Iso, g_loss_Sparse, g_loss_Conti = self.Iso_loss(xall_fake, xall_real, normal_org, self.dataset)
                           
                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_L1 * g_loss_L1 + self.lambda_Light * g_loss_Light + \
                            self.lambda_Iso * g_loss_Iso + self.lambda_Sparse * g_loss_Sparse + self.lambda_Conti * g_loss_Conti
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/fake'] = g_loss_fake.item()
                loss['G/L1'] = g_loss_L1.item()
                loss['G/Light'] = g_loss_Light.item()
                loss['G/Iso'] = g_loss_Iso.item()
                loss['G/Sparse'] = g_loss_Sparse.item()
                loss['G/Conti'] = g_loss_Conti.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                
                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)
  
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
            
            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+2) > self.num_iters_decay:
                g_lr = g_lr / 10
                d_lr = d_lr / 10
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):       
        """Predict normals using Photometric Stereo trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        data_loader = self.pixels_loader
        Dall = []
        
        d = os.path.join(self.result_dir,self.result_ind)+'/{}.txt'
        f = open(d.format(self.shape), 'w')

        with torch.no_grad():
            for i, (xall_real, x10_real, normal_org, ind) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x10_real = x10_real.to(self.device)
                xall_real = xall_real.to(self.device)
                normal_org = normal_org.to(self.device)
                ind = ind.to(self.device)
                                
                x_fake = self.G(x10_real)
                                    
                xdin_fake = torch.cat([x_fake, x10_real],dim=1)
                normal_fake = self.D(xdin_fake)
                delta = self.angular_deviation(normal_fake, normal_org, self.dataset)
                for j in range(delta.size(0)):
#                    f.write(str(normal_org[j].cpu().numpy()) + '\\' + str(normal_fake[j].cpu().numpy()) + '\\' + str(delta[j].cpu().numpy()) + '\\'  + str(ind[j].cpu().numpy())+'\t')
#                    f.write('\n')
                    Dall.append(delta[j])
                    
        f.write(str(np.mean(Dall)) + '\n')
        f.close()
        print('Dall=',np.mean(Dall))
        return np.mean(Dall)
