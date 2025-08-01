import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils
from test_MBA import eval_psnr
from models.mamber32_arch import  Mamba_ets,aMamba_ets
import math
import sys
import deg_utils
pca_matrix = torch.load(
        'pca_matrix.pth',
        map_location=lambda storage, loc: storage
    ).cuda()

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
#     if config.get('resume') is not None:
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        sv_file_ets = torch.load(config['resume_ets'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        ets = models.make(sv_file_ets['ets'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        optimizer_ets = utils.make_optimizer(
            ets.parameters(), sv_file_ets['optimizer_ets'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
            lr_scheduler_ets = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
            lr_scheduler_ets = MultiStepLR(optimizer_ets, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
            lr_scheduler_ets.step()


    else:
        model = models.make(config['model']).cuda()
        ets = models.make(config['ets']).cuda()

        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        optimizer_ets = utils.make_optimizer(
            ets.parameters(), config['optimizer'])

        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
            lr_scheduler_ets = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
            lr_scheduler_ets = MultiStepLR(optimizer_ets, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model,ets, optimizer,optimizer_ets, epoch_start, lr_scheduler,lr_scheduler_ets

import random
def train(train_loader, model,ets, optimizer,optimizer_ets, \
         epoch):
    model.train()
    ets.train()

    loss_fn = nn.L1Loss()
    loss_ets = nn.L1Loss()
    train_loss = utils.Averager()
    metric_fn = utils.calc_psnr

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    
    num_dataset = 800 # DIV2K
    iter_per_epoch = int(num_dataset / config.get('train_dataset')['batch_size'] \
                        * config.get('train_dataset')['dataset']['args']['repeat'])
    iteration = 0


    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()
        img=(batch['img']- inp_sub) / inp_div
        ref = (batch['ref'] - inp_sub) / inp_div
        inp = (batch['inp'] - inp_sub) / inp_div
        ref_hr = (batch['crop_ref_hr'] - inp_sub) / inp_div
        crop_ref = (batch['crop_ref'] - inp_sub) / inp_div
        ref_t=(batch['ref_txt']- inp_sub) / inp_div


        s = random.uniform(2, 4)
        prepro = deg_utils.SRMDPreprocessing(
            pca_matrix=pca_matrix, cuda=True, random_kernel=True,
            scale=s,
            ksize=21,
            code_length=10,
            sig_min=0.2,
            sig_max=4.0,
            rate_iso=1.0,
            random_disturb=False
        )

        # kernel shape: torch.Size([1, 21, 21])
        # ker map shape: torch.Size([1, 10])

        LR_img, ker_map, kernels = prepro(img,kernel=True)
        ker_map = torch.tensor(ker_map).cuda()
        kernels = torch.tensor(kernels).cuda()
        kernels=kernels.unsqueeze(dim=1)
        gen_lr = torch.tensor(LR_img).cuda()
        gen_ref=deg_utils.b_Bicubic(ref,s)
        pre_ker = ets(gen_lr, gen_ref)

        with torch.no_grad():
          ker=ets(inp, crop_ref)

        pred, ref_loss= model(inp, crop_ref,ker, ref_t, batch['coord'], batch['cell'])





        gt = (batch['gt'] - gt_sub) / gt_div
        # print(pred.shape, gt.shape)
        loss_pred = loss_fn(pred, gt)
        # print(kernels.shape,pre_ker.shape)
        loss_ker=loss_ets(ker_map,pre_ker)
        # print(loss_ker)



        psnr = metric_fn(pred, gt)


        loss_ref = loss_fn(ref_loss, ref_hr)

        loss = loss_pred * 0.7 + loss_ref * 0.3
        train_loss.add(loss.item())


        
        # tensorboard
        writer.add_scalars('loss', {'train': loss.item()}, (epoch-1)*iter_per_epoch + iteration)
        writer.add_scalars('psnr', {'train': psnr}, (epoch-1)*iter_per_epoch + iteration)
        iteration += 1



        optimizer_ets.zero_grad()
        loss_ker.backward()
        optimizer_ets.step()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None
        
    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model,ets, optimizer, optimizer_ets, epoch_start, lr_scheduler,lr_scheduler_ets = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model,ets, optimizer, optimizer_ets,\
                           epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()
            lr_scheduler_ets.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
#         writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
            ets_=ets.module
        else:
            model_ = model
            ets_=ets
        model_spec = config['model']
        ets_spec = config['ets']

        model_spec['sd'] = model_.state_dict()
        ets_spec['sd'] = ets_.state_dict()

        optimizer_spec = config['optimizer']
        optimizer_ets_spec = config['optimizer_ets']
        optimizer_spec['sd'] = optimizer.state_dict()
        optimizer_ets_spec['sd'] = optimizer_ets.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }
        sv_file_ets = {
            'ets': ets_spec,
            'optimizer_ets': optimizer_ets_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(sv_file_ets, os.path.join(save_path, 'ets_epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
            torch.save(sv_file_ets,
                       os.path.join(save_path, 'ets_epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
                ets_=ets.module
            else:
                model_ = model
                ets_=ets
            val_res = eval_psnr(val_loader, model_,ets_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
#             writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))
                torch.save(sv_file_ets, os.path.join(save_path, 'ets_epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='')
    parser.add_argument('--name', default='')
    parser.add_argument('--tag', default='')
    parser.add_argument('--gpu', default='')
    args = parser.parse_args()
    print(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('', save_name)
    
    main(config, save_path)
