import functools
import random
import math
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples



def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(torch.tensor(img))))
def fft2c(x):
    """
    x is a complex shapes [H,W,C]
    """
    S = x.shape
    x.reshape(S[0],S[1],-1)
    res = 1 / math.sqrt(S[0]*S[1]) * np.fft.fftshift(np.fft.fft2(x,axes=[0,1]),axes=[0,1])
    return res

def ifft2c(x):
    """
    x is a complex shapes [H,W,C]
    """
    S = x.shape
    x.reshape(S[0],S[1],-1)
    x = np.fft.ifftshift(x,axes=[0,1])
    res = math.sqrt(S[0]*S[1]) * np.fft.ifft2(x,axes=[0,1])
    return res

import numpy as np
from numpy.fft import fft2, fftshift
from numpy.fft import ifft2, ifftshift
def fft_2D(image,factor):
    k_space = fftshift(fft2(image))

    _,rows, cols = k_space.shape
    center_fraction = 1 / factor  # 中心区域比例
    mask = np.zeros((1,rows, cols))
    center_rows = int(rows * center_fraction // 2)
    center_cols = int(cols * center_fraction // 2)

    mask[:,rows // 2 - center_rows:rows // 2 + center_rows,
    cols // 2 - center_cols:cols // 2 + center_cols] = 1

    k_space_sampled = k_space * mask

    reconstructed_image = np.abs(ifft2(ifftshift(k_space_sampled)))
    return reconstructed_image



@register('ref-sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_hr,ref = self.dataset[idx]
        img_out=img_hr
        s = random.uniform(self.scale_min, self.scale_max)

        img = fft_2D(img_hr, factor=s)
        ref_kd=fft_2D(ref,factor=s)
        # img = img_hr
        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)]  # assume round int
            img_hr = img_hr[:, :round(h_lr * s), :round(w_lr * s)]
            ref_hr = ref[:, :round(h_lr * s), :round(w_lr * s)]
            ref_kd_lr=ref_kd[:, :round(h_lr * s), :round(w_lr * s)]

            img_down = resize_fn(img, (h_lr, w_lr))
            ref_down=resize_fn(ref_hr, (h_lr, w_lr))
            ref_kd_down=resize_fn(ref_kd_lr, (h_lr, w_lr))

            crop_lr, crop_hr,crop_ref,crop_ref_kd = img_down, img_hr,ref_down,ref_kd_down
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img_hr[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_ref = ref[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_ref_kd=ref_kd[:, x0: x0 + w_hr, y0: y0 + w_hr]

            # crop_hr=torch.from_numpy(crop_hr)

            crop_lr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_lr, w_lr)
            crop_ref=resize_fn(crop_ref, w_lr)
            crop_ref_kd = resize_fn(crop_ref_kd, w_lr)


            # crop_hr = img[:, x0: x0 + w_hr, :]
            # if crop_hr.shape[0]==3:
            #     crop_hr=torch.from_numpy(crop_hr.transpose(1,2,0))
            # crop_hr=torch.from_numpy(crop_hr)
            # crop_lr = resize_fn(crop_hr, (w_lr,crop_hr.shape[-1]))
            # print(crop_hr.shape)
            # print(crop_lr.shape)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                # if dflip:
                #     x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # crop_lr_up = F.interpolate(crop_lr.unsqueeze(0) , size=(crop_hr.shape[1], crop_hr.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
        # crop_res=crop_hr-crop_lr_up
        # _, ret_res_gt = to_pixel_samples(crop_res.contiguous())
        ref_t=torch.abs(crop_ref-crop_ref_kd)
        # print(torch.equal(crop_ref,crop_ref_kd))

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        _, T1_hr_rgb = to_pixel_samples(crop_ref.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            # print()
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            # ret_res_gt=ret_res_gt[sample_lst]

        ref_w = int(np.sqrt(T1_hr_rgb.shape[0]))
        ref_c = T1_hr_rgb.shape[1]
        last_ref_hr = T1_hr_rgb.view(ref_c, ref_w, ref_w)

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        # print(hr_rgb.shape)
        # print(crop_lr.shape)
        # print(hr_coord.unsqueeze(1).shape)
        # ret_res_gt=hr_rgb-F.grid_sample(crop_lr, hr_coord.flip(-1).unsqueeze(1), mode='bilinear', \
        #                      padding_mode='border', align_corners=False)[:, :, 0, :] \
        #     .permute(0, 2, 1)
        # #
        # torch.Size([8, 1, 64, 64])
        # torch.Size([8, 1, 4096, 2])
        # print('cell:',cell)
        s_formatted = "{:.2f}".format(s)
        s_tensor = torch.tensor(float(s_formatted))
        return {
            'inp': crop_lr,
            'ref':ref,
            "crop_ref_hr":last_ref_hr,
            'crop_ref':crop_ref,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            # 'res':ret_res_gt,
            'scale':s_tensor,
            'img': img_out,
            'ref_txt': ref_t,
            'ref_kd':ref_kd
        }



