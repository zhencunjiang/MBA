import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
import h5py
def normal(in_image):
    value_max = np.max(in_image)
    value_min = np.min(in_image)
    return (in_image - value_min) / (value_max - value_min)
def read_h5(file):
    f = h5py.File(file, 'r')
    x = f['image']
    x = np.array(x)
    # print(x.shape)
    # if len(x.shape)==2:
    #    x = np.expand_dims(x, axis=0)
    if np.max(x)>1:
        x=normal(x)

    return np.float32(x)

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'h5':
                self.files.append(transforms.ToTensor()(read_h5(file)))
            elif cache == 'h5_3':
                self.files.append(transforms.ToTensor()(read_h5(file)))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x
        elif self.cache == 'h5':
            # print('h5 shape:',x.size())

            return x
        elif self.cache == 'h5_3':
            # print('h5 shape:',x.size())
            [c,h,w]=x.shape
            x_3= np.random.randn(3,h,w)
            x_3[0,:,:]=x.squeeze(0)
            x_3[1,:, :] = x.squeeze(0)
            x_3[2,:, :] = x.squeeze(0)
            # print(x_3.shape)
            return x_3


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]

@register('ref-image-folder')
class ImageFolder2(Dataset):

    def __init__(self, root_path1,root_path2,
                 repeat=1, ):
        self.repeat = repeat

        filenames1 = sorted(os.listdir(root_path1))
        self.files1 = []
        for filename in filenames1:
            file = os.path.join(root_path1, filename)
            self.files1.append(transforms.ToTensor()(read_h5(file)))
        filenames2 = sorted(os.listdir(root_path2))
        self.files2 = []
        for filename in filenames2:
            file = os.path.join(root_path2, filename)
            self.files2.append(transforms.ToTensor()(read_h5(file)))

    def __len__(self):
        return len(self.files1) * self.repeat

    def __getitem__(self, idx):
        x1 = self.files1[idx % len(self.files1)]
        x2 = self.files2[idx % len(self.files2)]

        return x1,x2