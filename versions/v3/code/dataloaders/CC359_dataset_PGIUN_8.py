import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from os.path import splitext
from tqdm import tqdm
import torch.nn as nn
import h5py
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


class SliceData_CC359(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, 
                data_dir="/scratch/pf2m24/data/CCP359/Train/", 
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
        self.data_dir = data_dir
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.npy')]

        if rate < 1.0:
            all_files = all_files[:len(all_files)*rate]
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
        
        if mask_type == 'equispaced':
            
            self.mask = np.load('/scratch/pf2m24/projects/MambaIR/simple_mambair/data_loading/mask_0_256_af8.npy')
            print("mask shape", self.mask.shape)
            print("sum", self.mask[:,0,:].sum())
        self.coil_map = np.ones([1,256,256], dtype=np.float32)
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

        mask = self.mask
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
    
if __name__ == "__main__":
    dataset = SliceData_CC359()

        # Example usage
    from torch.utils.data import DataLoader

    val_loader = DataLoader(dataset=dataset,
                                  num_workers=0,
                                  batch_size=8,
                                  shuffle=True,)
    from tqdm import tqdm
    tbar = tqdm(val_loader, ncols=130, desc=f'Validate {0}')
    test_logs = []
    total_psnr = total_ssim = total_rmse = 0 
    for batch_idx, batch in enumerate(tbar):

        # images, true_masks = batch[0].to(device), batch[1].to(device)
        inputs, target, masked_kspace, mask, fname, slice_id = batch
        total_psnr += peak_signal_noise_ratio(target.detach().cpu().numpy(), inputs.detach().cpu().numpy(), data_range=target.max())
        total_rmse += nmse(target.detach().cpu().numpy(), inputs.detach().cpu().numpy())
        total_ssim += calculate_ssim(target.detach().cpu().numpy(), inputs.detach().cpu().numpy()) 

        test_logs.append({
            'fname': fname,
            'slice': slice_id,
            'output': (inputs).squeeze(1).cpu().detach().numpy(),
            'target': (target).squeeze(1).cpu().numpy(),
        })
        
    from collections import defaultdict
    outputs = defaultdict(list)
    targets = defaultdict(list)
    for log in test_logs:
        for i, (fname, slice_id) in enumerate(zip(log['fname'], log['slice'])):
            outputs[fname].append((slice_id, log['output'][i]))
            targets[fname].append((slice_id, log['target'][i]))

    metrics = dict(nmse=[], ssim=[], psnr=[])

    for fname in tqdm(outputs):
        output = np.stack([out for _, out in sorted(outputs[fname])])
        target = np.stack([tgt for _, tgt in sorted(targets[fname])])
        metrics['nmse'].append(nmse(target, output))
        metrics['ssim'].append(re_ssim(target, output))
        metrics['psnr'].append(re_psnr(target, output))
        
    metrics = {metric: np.mean(values) for metric, values in metrics.items()}
    print("reconformer paitient", metrics, '\n')
    print('No. Volume: ', len(outputs))
    print("reconformer slice", total_psnr/len(val_loader), total_ssim/len(val_loader))
