# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:18:19 2018

@author: jiaym15
"""

import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import numpy as np

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    pixels_loader= None

    if config.mode == 'train':
        pixels_loader = get_loader(config.train_image_dir, config.test_image_dir, config.image_size, 
                           config.batch_size, 'Pixels', config.mode, config.num_workers)
        solver = Solver(pixels_loader, config)
        solver.train()
    elif config.mode == 'test':
        shape_list=['ballPNG','bearPNG','buddhaPNG','catPNG','cowPNG','gobletPNG','harvestPNG','pot1PNG','pot2PNG','readingPNG']
#        shape_list=['buddhaPNG','catPNG','horsePNG','owlPNG','sheepPNG']
        for i in range(100):
            config.result_ind = str(i+1)
            f = open(os.path.join('photometric/results',str(i+1))+'.txt','w')
            if not os.path.exists(os.path.join(config.result_dir,str(i+1))):
                os.makedirs(os.path.join(config.result_dir,str(i+1)))
            for j in range(len(shape_list)):
                config.shape = shape_list[j]
                test_dir=os.path.join(config.test_image_dir,str(i+1),config.shape)
                pixels_loader = get_loader(config.train_image_dir, test_dir, config.image_size, 
                   config.batch_size, 'Pixels', config.mode, config.num_workers)
                solver = Solver(pixels_loader, config)
                delta = solver.test()
                f.write(str(delta) + '\n')
                del pixels_loader
            f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--image_size', type=int, default=32, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=5, help='number of strided conv layers in D')
    parser.add_argument('--lambda_L1', type=float, default=180/np.pi, help='weight for L1 loss')
    parser.add_argument('--lambda_Light', type=float, default=180/np.pi, help='weight for Light loss')
    parser.add_argument('--lambda_Iso', type=float, default=0.02*180/np.pi, help='weight for Iso loss')
    parser.add_argument('--lambda_Sparse', type=float, default=2e-5*180/np.pi, help='weight for Sparse loss')
    parser.add_argument('--lambda_Conti', type=float, default=1e-3*180/np.pi, help='weight for Continuity loss')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='Pixels',help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    # Directories.
    parser.add_argument('--train_image_dir', type=str, default='data/train')
    parser.add_argument('--test_image_dir', type=str, default='data/test')
    parser.add_argument('--shape', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='photometric/logs')
    parser.add_argument('--model_save_dir', type=str, default='photometric/models')
    parser.add_argument('--result_dir', type=str, default='photometric/results')
    parser.add_argument('--result_ind', type=str, default=None)
    
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=50000)

    config = parser.parse_args()
    print(config)
    main(config)
