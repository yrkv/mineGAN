import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
# from operation import ImageFolder
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

import os
import time
import random
import string
import argparse
import itertools
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import model
# from diffaug import DiffAugment
# policy = 'color,translation'
import lpips
#TODO: try other perceptual losses
percept = lpips.LPIPS(net='vgg')

criterion = nn.BCELoss()
model_config_args = {
    'nc', 'nz', 'ngf', 'ndf', 'lr', 'beta1',

    'd_dropout',
    #'d_noise',

    #'g_dropout',
    'g_up_block',
}


#def set_dropout_p(model, p):
#    def to_apply(m):
#        classname = m.__class__.__name__
#        if classname.find('Dropout') != -1:
#            m.p = p
#    model.apply(to_apply)

#small_quarter_parts = [
#    (slice(None,8), slice(None,8)),
#    (slice(None,8), slice(8,None)),
#    (slice(8,None), slice(None,8)),
#    (slice(8,None), slice(8,None)),
#]
#
#def scale_slice(in_slice, from_scale=8, to_scale=64):
#    start, stop = in_slice.start, in_slice.stop
#    if start is not None:
#        start = start * to_scale // from_scale
#    if stop is not None:
#        stop = stop * to_scale // from_scale
#    return slice(start, stop)
#
#def get_quarter_parts(hw=256, part=None):
#    if part is None:
#        part = random.randint(0, 3)
#
#    small_part = small_quarter_parts[part]
#    big_part = scale_slice(small_part[0]), scale_slice(small_part[1])
#    return small_part, big_part

def get_part(image, part, k, to_size=64):
    from_size = 512 * 8 // k

    selected = image[:,:,0:from_size,0:from_size]

    return F.interpolate(selected, to_size)


def near_one_like(input, rand_range=0.1):
    return 1 - torch.rand_like(input)*rand_range

def near_zero_like(input, rand_range=0.1):
    return torch.rand_like(input)*rand_range

def train_d(net, data, label="real"):
    #small_part, big_part = get_quarter_parts()
    pred, *rec = net(data, label=label, part=0)

    if pred.dim() > 1:
        pred_mean = pred.mean(list(range(1, pred.dim()))).round()
    else:
        pred_mean = pred.round()

    if label=="real":
        target = near_zero_like(pred)
        target_mean = 0
        #print(rec[0].shape, rec[1].shape, rec[2].shape)
        #print(small_part, big_part)
        #data_part = data[:,:,big_part[0], big_part[1]]
        #print(data_part.shape)
        #err = criterion(pred, target) + 0.1*( \
                #percept(rec[0], F.interpolate(data, rec[0].size()[-2:])).sum() + \
                #percept(rec[1], F.interpolate(data, rec[1].size()[-2:])).sum() + \
                #percept(rec[2], F.interpolate(data[:,:,big_part[0],big_part[1]], rec[2].size()[-2:])).sum())
        err_d = criterion(pred, target)
        err_rec = percept(rec[0], get_part(data, 0, 8)).sum() + \
                percept(rec[1], get_part(data, 0, 16)).sum() + \
                percept(rec[2], get_part(data, 0, 32)).sum() + \
                percept(rec[3], get_part(data, 0, 64)).sum()
        err = err_d / err_d.sum().detach() + err_rec / err_rec.sum().detach()
    else:
        target = near_one_like(pred)
        target_mean = 1
        err = criterion(pred, target)

    err.backward()

    p_correct = (pred_mean == target_mean).float().mean()
    return err.mean().item(), p_correct


def load_checkpoint(ckpt, ckpt_dir):
    if ckpt is None:
        return None

    if ckpt == 'last':
        search_str = os.path.join(ckpt_dir, 'minetestgan_*.pt')
        found_checkpoints = glob(search_str)
        if len(found_checkpoints) == 0:
            return None
        load_checkpoint = max(found_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f'using checkpoint {load_checkpoint}')

    return torch.load(load_checkpoint)


def train(args):
    # config -- model configuration details
    config = {k:args[k] for k in model_config_args}
    device = args['device']
    name = args['name']
    if args['name'] is None:
        name = ''.join(np.random.choice(list(string.ascii_lowercase), size=10, replace=False))
    save_dir = os.path.join(args['results_dir'], name)

    percept.to(device)

    # make sure output directories exist
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)

    
    #TODO: custom data loader
    dataset = ImageFolder(root=args['data_dir'], transform=transforms.Compose([
#         transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]))
    dataloader = DataLoader(dataset, batch_size=args['batch_size'],
                            shuffle=True, num_workers=2)
    
    netG = model.Generator(config).to(device)
    netD = model.Discriminator(config).to(device)
    optG = optim.Adam(netG.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
    optD = optim.Adam(netD.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))

    checkpoint = load_checkpoint(args['ckpt'], os.path.join(save_dir, 'checkpoints'))
    if checkpoint is not None:
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optG.load_state_dict(checkpoint['optG_state_dict'])
        optD.load_state_dict(checkpoint['optD_state_dict'])
        start_epoch = checkpoint['epoch']
        fixed_noise = checkpoint['fixed_noise']
    else:
        start_epoch = 0
        fixed_noise = torch.randn(64, config['nz'], device=device)

    
    steps = 0

    for epoch in range(start_epoch, args['epochs']):
        epoch_start_time = time.time()
        epoch_log = []

        data_iter = enumerate(dataloader)
        if args['progress']:
            data_iter = tqdm(data_iter, total=len(dataloader))
            
        for i, data in data_iter:
            steps += 1
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            noise = torch.randn(b_size, config['nz'], device=device)

            fake_images = netG(noise)
            #TODO: test DiffAugment
#             real_images = DiffAugment(real_images, policy=policy)
#             fake_images = DiffAugment(fake_images, policy=policy)

            # train Discriminator
            netD.zero_grad()
            err_dr, c_dr = train_d(netD, real_images, label="real")
            err_df, c_df = train_d(netD, fake_images.detach(), label="fake")
            optD.step()

            # train Generator
            netG.zero_grad()
            pred_g, = netD(fake_images)
#             err_g = -pred_g.mean()
            target = near_zero_like(pred_g)
            err_g = criterion(pred_g, target)
            err_g.backward()
            optG.step()


            epoch_log.append({
                'err_dr': err_dr,
                'c_dr': c_dr,
                'err_df': err_df,
                'c_df': c_df,
                'err_g': err_g.item(),
            })

        
        df = pd.DataFrame(epoch_log)
        err_dr = df['err_dr'].mean()
        c_dr = df['c_dr'].mean()
        err_df = df['err_df'].mean()
        c_df = df['c_df'].mean()
        err_g = df['err_g'].mean()
        t = time.time() - epoch_start_time
        print(f'Epochs: {epoch+1}, {t=:.1f}, '
                f' {err_dr=:.4f}, {c_dr=:.4f}, '
                f' {err_df=:.4f}, {c_df=:.4f}, '
                f' {err_g=:.4f}')

        with torch.no_grad():
            im = to_pil_image(make_grid(netG(fixed_noise), normalize=True, value_range=(-1,1)))
            image_path = os.path.join(save_dir, f'images/fixed_images_{epoch+1:04}.png')
            im.save(image_path)

        # checkpoint every (--ckpt_every) epochs
        if args['ckpt_every'] != 0 and (epoch+1) % args['ckpt_every'] == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoints/minetestgan_{epoch+1:04}.pt')
            torch.save({
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optG_state_dict': optG.state_dict(),
                'optD_state_dict': optD.state_dict(),
                'epoch': epoch+1,
                'fixed_noise': fixed_noise,
                'config': config,
            }, checkpoint_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='minetest gan')

    parser.add_argument('--batch_size', type=int, default=128, help='mini batch number of images')
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--ckpt', type=str, default='last',
            help='checkpoint weight path, or \'last\' to use most recent. To '
            'use checkpointing, provide a --name or a random one is generated.')
    parser.add_argument('--ckpt_every', type=int, default=0,
            help='checkpoint every N epochs. If 0, disable checkpointing.')
    parser.add_argument('--results_dir', type=str, default='train_results',
            help='directory dump train results and search for checkpoints')
    parser.add_argument('--name', type=str, default=None, help='experiment name (default: randomly generated)')
    parser.add_argument('--data_dir', type=str, default='minetest-data', help='train images directory')
    parser.add_argument('--progress', action='store_true', help='Add progress bars to individual epochs')
    parser.add_argument('--epochs', type=int, default=100, help='')

    parser.add_argument('--nc', type=int, default=3, help='')
    parser.add_argument('--nz', type=int, default=100, help='')
    parser.add_argument('--ngf', type=int, default=32, help='')
    parser.add_argument('--ndf', type=int, default=32, help='')
    parser.add_argument('--lr', type=float, default=0.0002, help='')
    parser.add_argument('--beta1', type=float, default=0.5, help='')

    parser.add_argument('--d_dropout', type=float, default=0.0, help='')
    #parser.add_argument('--d_noise', type=float, default=0.0, help='')

    #parser.add_argument('--g_dropout', type=float, default=0.0, help='')
    parser.add_argument('--g_up_block', type=str, default='UpBlock', help='')


#     'BN_momentum': 0.8,
#     'noisy_input_D': 0.5,
#     'noisy_input_D_gamma': 0.8,
#     'leakyReLU_slope': 0.2,
#     'dropout_G': 0.2,
#     'dropout_D': 0.2,
#     'dropout_G_gamma': 0.8,
#     'dropout_D_gamma': 0.8,


    args = parser.parse_args()
    print(args)
    train(vars(args))
