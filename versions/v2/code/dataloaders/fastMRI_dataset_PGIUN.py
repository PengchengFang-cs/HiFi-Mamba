import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from os.path import splitext
from tqdm import tqdm
import torch.nn as nn
import h5py
from dataloaders.subsample import create_mask_for_mask_type
#from subsample import create_mask_for_mask_type
from torchvision.utils import save_image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import random

class fastMRI_dataset(Dataset):

    def __init__(self, data_dir, select='FSPD', type='train', acceleration=4, mask_type = 'random', resolution=192, rate=1.0):
        super(fastMRI_dataset, self).__init__()
        self.data_dir = data_dir
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]
        if rate < 1.0:
            all_files = all_files[:len(all_files)*rate]
        files = []
        self.examples = []
        for file in tqdm(all_files):
            path_file = os.path.join(self.data_dir, file)
            with h5py.File(path_file, 'r') as f:
                acq = f.attrs["acquisition"]
                if select == 'PD':
                    if acq == "CORPD_FBK":
                        files.append(path_file)
                else:
                    if acq != "CORPD_FBK":
                        files.append(path_file)

        for fname in tqdm(sorted(files)):
            data = h5py.File(fname, 'r')
            rss_images = np.array(data["reconstruction_rss"])
            num_slices = rss_images.shape[0]
            num_start = int(num_slices / 2 -10)
            num_end = num_start + 20
            self.examples += [(fname, slice_id) for slice_id in range(num_start, num_end)]

        self.acceleration = [acceleration]
        self.type = type
        center_fractions = [0.32 / acceleration]  # Adjusted for acceleration
        if mask_type == 'equispaced':
            self.mask = np.load('/scratch/pf2m24/projects/MambaIR/simple_mambair/data_loading/mask_0_320.npy')
            #self.mask = np.load('/scratch/pf2m24/projects/MambaIR/simple_mambair/data_loading/mask_2_320_af4.npy')
            print("mask shape", self.mask.shape)
            print("sum", self.mask[:,0,:].sum())
        self.mask_type = mask_type
        mask_func = create_mask_for_mask_type(mask_type, center_fractions,
                                    self.acceleration)
        self.resolution = resolution
        self.data_trans = DataTransform(
            resolution=resolution,
            mask_func=mask_func,  # Replace with actual mask function if needed
            use_seed=True
        )
        self.coil_map = np.ones([1,320,320], dtype=np.float32)

    def read_datafile(self, file_path, slice_index):
        data = h5py.File(file_path, 'r')
        rss_images = np.array(data["reconstruction_rss"])[slice_index]
        return rss_images

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fname, slice_id = self.examples[idx]
        image_rec = self.read_datafile(fname, slice_id)
        image_rec = self.norm(image_rec)
        
        fully = np.fft.fftshift(np.fft.fft2(image_rec))
        fully = np.expand_dims(fully, axis=0)
        if self.mask_type == 'equispaced':
            mask = self.mask
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
        # return (
        #         under_image_rec,
        #         image_rec, # gt
        #         under_sampling,#.to(torch.float32), # undersample kspace
        #         mask,
        #         fname,
        #         slice_id,
        # )
    def norm(self, image_2D):
        max_ = np.max(image_2D)
        min_ = np.min(image_2D)
        if max_ == 0:
            return image_2D
        return (image_2D - min_) / (max_ - min_)
    
class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution, mask_func=None, use_seed=True):
  
        self.mask_func = mask_func
        self.resolution = resolution
        self.use_seed = use_seed

    def __call__(self, fname):
        # Apply mask
        shape = np.array((1, self.resolution, self.resolution))
        shape[:-3] = 1
        seed = None if not self.use_seed else tuple(map(ord, fname))
        mask = self.mask_func(shape, seed).reshape(1,1,self.resolution)
        mask = mask.repeat(1,shape[1], 1).squeeze().unsqueeze(0)
        return mask
    
def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    
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

if __name__ == "__main__":
    path = '/scratch/pf2m24/data/FastMRI/singlecoil_val'
    dataset = fastMRI_dataset(data_dir=path, select='PD', type='train', acceleration=8, mask_type='equispaced')
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
