from models.Discriminator import Discriminator
from models.Generator import Generator

from src.utils import objectview

from src.functions import train, validate, LinearLrDecay, load_params, copy_params, cur_stages
from src.utils import set_log_dir, save_checkpoint, create_logger
from src.metrics.inception_score import _init_inception
from src.metrics.fid_score import create_inception_graph, check_or_download_inception

import torch
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from adamw import AdamW

import datasets

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


options = {
    'init_type':'normal',
    'mask':8_8,
    'gf_dim':1024,
    'df_dim':,
    'latent_dim':1024,
    'd_depth':5,
    'patch_size':1,
    'img_size':32,
    'diff_aug':'translation',
    'optimizer': 'adamw',
    'lr_G':1e-5,
    'lr_D':1e-5,
    'wd':1e-3,
    'beta1':0,
    'beta2':0.99,
    'max_iter':100000,
    'max_epoch':200,
    'output_dir':'ckpt',
    'n_critic':5,
    'dataset':'cifar10',
    'data_path':'.',
    'num_workers':36,
    'dis_batch_size':100,
    'fad_in':0
}

options=objectview(options)


netG = Generator(options)
netG = Discriminator(options)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if options.init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif options.init_type == 'orth':
            nn.init.orthogonal_(m.weight.data)
        elif options.init_type == 'xavier_uniform':
            nn.init.xavier_uniform(m.weight.data, 1.)
        else:
            raise NotImplementedError('{} unknown inital type'.format(options.init_type))
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

netG.apply(weights_init)
netD.apply(weights_init)

gpu_ids = [i for i in range(int(torch.cuda.device_count()))]
netG = torch.nn.DataParallel(netG.to("cuda:0"), device_ids=gpu_ids)
netD = torch.nn.DataParallel(netD.to("cuda:0"), device_ids=gpu_ids)

if options.optimizer == "adam":
    optimizerG = torch.optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()),
                                    options.lr_G, (options.beta1, options.beta2))
    optimizerD = torch.optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()),
                                    options.lr_D, (options.beta1, options.beta2))
elif options.optimizer == "adamw":
    optimizerG = AdamW(filter(lambda p: p.requires_grad, netG.parameters()),
                                    options.lr_G, weight_decay=options.wd)
    optimizerD = AdamW(filter(lambda p: p.requires_grad, netD.parameters()),
                                     options.lr_D, weight_decay=options.wd)
    
schedulerG= LinearLrDecay(gen_optimizer, options.lr_G, 0.0, 0, options.max_iter * options.n_critic)
schedulerD = LinearLrDecay(dis_optimizer, options.lr_D, 0.0, 0, options.max_iter * options.n_critic)


dataset = datasets.ImageDataset(options, cur_img_size=8)
train_loader = dataset.train

for epoch in range(options.max_epoch):
        train(options, gen_net = netG, dis_net = netD, gen_optimizer = optimizerG, dis_optimizer = optimizerD, gen_avg_param = None, train_loader = train_loader,
            epoch = epoch, writer_dict = writer_dict, fixed_z = None, schedulers=[schedulerG, schedulerD])

        checkpoint = {'epoch':epoch, 'best_fid':best}
        checkpoint['gen_state_dict'] = gen_net.state_dict()
        checkpoint['dis_state_dict'] = dis_net.state_dict()
        score = validate(options, None, fid_stat, epoch, gen_net, writer_dict, clean_dir=True)
        # print these scores, is it really the latest
        print(f'FID score: {score} - best ID score: {best} || @ epoch {epoch}.')
        if epoch == 0 or epoch > 30:
            if score < best:
                save_checkpoint(checkpoint, is_best=(score<best), output_dir=options.output_dir)
                print("Saved Latest Model!")
                best = score

    checkpoint = {'epoch':epoch, 'best_fid':best}
    checkpoint['gen_state_dict'] = gen_net.state_dict()
    checkpoint['dis_state_dict'] = dis_net.state_dict()
    score = validate(options, None, fid_stat, epoch, gen_net, writer_dict, clean_dir=True)
    save_checkpoint(checkpoint, is_best=(score<best), output_dir=options.output_dir)