import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from os.path import splitext
from tqdm import tqdm
import torch.nn as nn
import h5py
from dataloaders.subsample import create_mask_for_mask_type
from torchvision.utils import save_image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import random

def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def re_psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())

def re_ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(
        #gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=False, data_range=pred.max() - pred.min()
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range = gt.max()
    )

def calculate_ssim(img1, img2):
    """计算四维张量 (batch_size, channels, width, height) 的 SSIM"""
    ssim_values = []
    batch_size, channels, width, height = img1.shape
    for i in range(batch_size):
        for j in range(channels):
            ssim_value = structural_similarity(img1[i, j], img2[i, j], data_range=img2[i, j].max()) #- img2[i, j].min())
            ssim_values.append(ssim_value)
    return np.mean(ssim_values)


class DataTransform:
    def __init__(self, resolution, mask_func=None, use_seed=True):
        self.mask_func = mask_func
        self.resolution = resolution
        self.use_seed = use_seed

    def __call__(self, fname):
        shape = np.array((1, self.resolution, self.resolution))
        shape[:-3] = 1
        seed = None if not self.use_seed else tuple(map(ord, fname))
        mask = self.mask_func(shape, seed).reshape(1, 1, self.resolution)
        mask = mask.repeat(1, shape[1], 1).squeeze().unsqueeze(0)
        return mask


class SliceData_CC359(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, 
                data_dir=None,
                select='FSPD', type='train', acceleration=8, mask_type = 'equispaced', resolution=320, rate=1.0):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if data_dir is None:
            raise ValueError("data_dir must be provided.")
        self.data_dir = data_dir
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.npy')]

        if rate < 1.0:
            all_files = all_files[:int(len(all_files) * rate)]
        self.examples = []
        for fname in sorted(all_files):
            kspace = np.load(fname).transpose(0,3,1,2)

            padding_left = None
            padding_right = None

            num_slices = kspace.shape[0]
            num_start = 30#int(num_slices / 2 -10)
            num_end = num_slices - 30

            self.examples += [(fname, slice, padding_left, padding_right) for slice in range(num_start, num_end)]#-5)]
        print("data num:", len(self.examples))
        
        self.acceleration = [acceleration]
        self.mask_type = mask_type
        center_fractions = [0.32 / acceleration]
        mask_func = create_mask_for_mask_type(mask_type, center_fractions, self.acceleration)
        self.resolution = resolution
        self.data_trans = DataTransform(
            resolution=resolution,
            mask_func=mask_func,
            use_seed=True,
        )
        self.mask = None
        if mask_type == 'equispaced':
            self.mask = self.data_trans(f"fixed_{mask_type}_{acceleration}_{resolution}").numpy()
        self.coil_map = np.ones([1, resolution, resolution], dtype=np.float32)
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, padding_left, padding_right = self.examples[i]
        fully = np.load(fname)[slice].transpose(2,0,1).astype(np.float32)
        fully = fully[0] + 1j*fully[1]
        fully = np.fft.ifftshift(fully)
        image_rec = np.abs(np.fft.ifft2(np.fft.ifftshift(fully)))
        image_rec = self.norm(image_rec)
        fully = np.fft.fftshift(np.fft.fft2(image_rec))
        fully = np.expand_dims(fully, axis=0)

        if self.mask is not None:
            mask = self.mask.copy()
        else:
            mask = self.data_trans(fname).numpy()
        under_sampling = fully * mask
        under_image_rec = np.fft.ifft2(np.fft.ifftshift(under_sampling))
        # under_image_rec  = np.abs(under_image_rec)

        under_image_rec = np.stack([under_image_rec.real, under_image_rec.imag], axis=0).squeeze()
        image_rec = np.expand_dims(image_rec, axis=0)
        # image_rec = np.abs(image_rec)
        image_rec = np.stack([image_rec.real, image_rec.imag], axis=0).squeeze()

        image_rec = torch.from_numpy(np.ascontiguousarray(image_rec)).to(torch.float32)
        under_image_rec = torch.from_numpy(np.ascontiguousarray(under_image_rec)).to(torch.float32)
        under_sampling = np.stack([under_sampling.real, under_sampling.imag], axis=0)

        # under_sampling = torch.from_numpy(under_sampling).to(torch.complex64)
        under_sampling = torch.from_numpy(under_sampling).to(torch.float32)
        mask = torch.from_numpy(mask).to(torch.float32)

        return dict(us_image=under_image_rec, fs_image=image_rec, us_mask=mask, coil_map=torch.from_numpy(self.coil_map))
        
    
    def norm(self, image_2D):
        max_ = np.max(image_2D)
        min_ = np.min(image_2D)
        if max_ == 0:
            return image_2D
        return (image_2D - min_) / (max_ - min_)
    
