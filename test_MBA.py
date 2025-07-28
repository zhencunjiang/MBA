import argparse
import os
import math
from functools import partial
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
import torch.nn as nn
from models.mamber32_arch import  Mamba_ets,single_Mamba_ets,aMamba_ets


def batched_predict(model, inp,ker,cropref, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp,ker,cropref)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred
from PIL import Image
import numpy as np
def save_tensor_as_image(tensor, file_path):
    image_tensor = tensor.squeeze(0).squeeze(0)
    # 将 Tensor 转换为 NumPy 数组
    image_array = image_tensor.cpu().numpy()
    # 如果 Tensor 数据范围是 [0, 1]，需要归一化到 [0, 255] 并转换为 uint8 类型
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        # 如果数据范围超出了 [0, 1]，直接转换为 uint8 类型
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    # 将 NumPy 数组转换为 PIL 图像
    image = Image.fromarray(image_array)
    # 保存图像

    image.save(file_path)

import matplotlib.pyplot as plt

def save_residual_map(gt_tensor, recon_tensor, file_path, output_size=(256, 256)):
    # 移除 batch 和 channel 维度：[1, 1, H, W] → [H, W]
    gt_np = gt_tensor.squeeze().detach().cpu().numpy()
    recon_np = recon_tensor.squeeze().detach().cpu().numpy()

    # 计算残差图并归一化
    residual = np.abs(gt_np - recon_np)
    residual = residual / (np.max(residual) + 1e-8)

    # 临时保存路径（PNG格式支持Alpha通道）
    temp_path = file_path + ".temp.png"

    # 绘图并保存为 PNG
    plt.figure(figsize=(2.56, 2.56), dpi=100)
    im = plt.imshow(residual, cmap='bwr', vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.margins(0, 0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    image = Image.open(temp_path).resize(output_size, Image.BILINEAR)
    image = image.convert("RGB")  # 添加这一行修复 RGBA 问题
    image.save(file_path)



def eval_psnr(loader, model,ets, data_norm=None, eval_type=None, eval_bsize=None, window_size=0, scale_max=4, fast=False,
              verbose=False):
    model.eval()
    ets.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        print(scale)
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')


    number=0
    test_SSIM = 0
    test_PSNR = 0
    list_p=[]
    list_s=[]
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()
        crop_ref = (batch['crop_ref']- inp_sub) / inp_div
        inp = (batch['inp'] - inp_sub) / inp_div
        ref_t=(batch['ref_txt']- inp_sub) / inp_div


        # ker = ets(inp)

        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]
            
            coord = utils.make_coord((scale*(h_old+h_pad), scale*(w_old+w_pad))).unsqueeze(0).cuda()
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0
            
            coord = batch['coord']
            cell = batch['cell']


        with torch.no_grad():

                ker = ets(inp,crop_ref)
                pred,_ = model(inp,crop_ref,ker, ref_t, coord, cell)

            
        pred = pred * gt_div + gt_sub
        inp_out=inp*inp_div+inp_sub


        pred.clamp_(0, 1)

        if eval_type is not None and fast == False: # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 1]
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            
            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 1]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]
            
        res = metric_fn(pred, batch['gt'])
        p=PSNR(pred.cpu().numpy(), batch['gt'].cpu().numpy(),data_range=1)
        # print(pred.cpu().numpy().shape, batch['gt'].cpu().numpy().shape)
        s = SSIM(pred[0][0].cpu().numpy(), batch['gt'][0][0].cpu().numpy(), data_range=1)
        test_PSNR += p
        test_SSIM+=s
        list_p.append(s)
        list_s.append(p)
        val_res.add(res.item(), inp.shape[0])

        number+=1
        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    print(test_PSNR/number)
    print(test_SSIM/number)


    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='')
    parser.add_argument('--model',
                        default='')
    parser.add_argument('--ets',
                        default='')


    parser.add_argument('--window', default='0')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--fast', default=False)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=0, pin_memory=True,shuffle=False)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    ets_spec = torch.load(args.ets)['ets']
    ets = models.make(ets_spec, load_sd=True).cuda()


    res = eval_psnr(loader, model,ets,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        window_size=int(args.window),
        scale_max = args.scale_max,
        fast = args.fast,
        verbose=True)
    print('result: {:.4f}'.format(res))

