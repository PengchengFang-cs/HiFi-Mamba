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
from glob import glob
import nibabel
import SimpleITK as sitk

sitk.ProcessObject_SetGlobalWarningDisplay(False)

def list_nii_files(path):
    nii_files = []
    for entry in os.scandir(path):
        if entry.is_dir():
            # 进入子目录扫描
            for f in os.scandir(entry.path):
                if (
                    f.is_file() 
                    and f.name.endswith(".nii.gz") 
                    and not f.name.endswith("_4d.nii.gz")
                    and not f.name.endswith("_gt.nii.gz")
                ):
                    x = nibabel.load(f.path)
                    shape = x.get_fdata().shape
                    for i in range(shape[-1]):
                        nii_files.append((f.path, i))
    return nii_files


def center_crop_128(arr: np.ndarray) -> np.ndarray:
    """
    从任意 2D numpy 数组中裁出中心的 128×128 区域。
    若原图比 128 小，会自动补零。
    """
    if arr.ndim != 2:
        raise ValueError("输入必须是 2D numpy 数组")

    h, w = arr.shape
    target = 128

    # 如果原始尺寸比128小，则先补零到至少128×128
    if h < target or w < target:
        pad_h = max(0, (target - h + 1) // 2)
        pad_w = max(0, (target - w + 1) // 2)
        arr = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        h, w = arr.shape

    # 计算中心裁剪起点
    top = (h - target) // 2
    left = (w - target) // 2

    # 中心裁剪
    return arr[top:top + target, left:left + target]

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

class SliceData_ACDC(Dataset):
    def __init__(self, 
                data_dir="/scratch/pf2m24/data/ACDC/training", mask_path="/scratch/pf2m24/projects/MRIRecon/MambaReconV3/code/dataloaders/radial_mask/radial_mask_32.npy"):
        self.all_files = list_nii_files(data_dir)
        self.mask = np.load(mask_path)

        print("data num:", len(self.all_files))

        self.coil_map = np.ones([1,128,128], dtype=np.float32)
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, i):
        fname, slice_num = self.all_files[i]
        dataA = sitk.ReadImage(fname)
        dataA = sitk.GetArrayFromImage(dataA).astype(np.float32)
        dataA = center_crop_128(dataA[slice_num])
        image_rec = (dataA - dataA.min()) / (dataA.max() - dataA.min())
        k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image_rec)))

        
        k = k * self.mask
        under_image_rec = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(k)))
        # save_image(torch.from_numpy(np.abs(under_image_rec)).unsqueeze(0).unsqueeze(0), f"under_image_rec_{i}.png")
        under_image_rec = np.stack([under_image_rec.real, under_image_rec.imag], axis=0).squeeze()

        under_image_rec = torch.from_numpy(under_image_rec).to(torch.float32)
        mask = torch.from_numpy(self.mask).to(torch.float32).unsqueeze(0)

        image_rec = np.stack([image_rec.real, image_rec.imag], axis=0).squeeze()
        image_rec = torch.from_numpy(image_rec).to(torch.float32)

        return dict(us_image=under_image_rec, fs_image=image_rec, us_mask=mask, coil_map=torch.from_numpy(self.coil_map))

class Unet_ACDC(Dataset):
    def __init__(self, 
                data_dir="/scratch/pf2m24/data/ACDC/training", mask_path="/scratch/pf2m24/projects/MRIRecon/MambaReconV3/code/dataloaders/radial_mask/radial_mask_32.npy"):
        self.all_files = list_nii_files(data_dir)
        self.mask = np.load(mask_path)

        print("data num:", len(self.all_files))

        self.coil_map = np.ones([1,128,128], dtype=np.float32)
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, i):
        fname, slice_num = self.all_files[i]
        dataA = sitk.ReadImage(fname)
        dataA = sitk.GetArrayFromImage(dataA).astype(np.float32)
        dataA = center_crop_128(dataA[slice_num])
        image_rec = (dataA - dataA.min()) / (dataA.max() - dataA.min())
        k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image_rec)))

        
        k = k * self.mask
        under_image_rec = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(k)))
        # save_image(torch.from_numpy(np.abs(under_image_rec)).unsqueeze(0).unsqueeze(0), f"under_image_rec_{i}.png")
        under_image_rec = np.abs(under_image_rec)

        under_image_rec = torch.from_numpy(under_image_rec).to(torch.float32).unsqueeze(0)
        mask = torch.from_numpy(self.mask).to(torch.float32).unsqueeze(0)

        image_rec = np.abs(image_rec)
        image_rec = torch.from_numpy(image_rec).to(torch.float32).unsqueeze(0)

        return dict(us_image=under_image_rec, fs_image=image_rec, us_mask=mask, coil_map=torch.from_numpy(self.coil_map))

class Ista_ACDC(Dataset):
    def __init__(self, 
                data_dir="/scratch/pf2m24/data/ACDC/training", mask_path="/scratch/pf2m24/projects/MRIRecon/MambaReconV3/code/dataloaders/radial_mask/radial_mask_32.npy"):
        self.all_files = list_nii_files(data_dir)
        self.mask = np.load(mask_path)

        print("data num:", len(self.all_files))

        self.coil_map = np.ones([1,128,128], dtype=np.float32)
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, i):
        fname, slice_num = self.all_files[i]
        dataA = sitk.ReadImage(fname)
        dataA = sitk.GetArrayFromImage(dataA).astype(np.float32)
        dataA = center_crop_128(dataA[slice_num])
        image_rec = (dataA - dataA.min()) / (dataA.max() - dataA.min())
        k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image_rec)))

        
        k = k * self.mask
        under_image_rec = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(k)))
        # save_image(torch.from_numpy(np.abs(under_image_rec)).unsqueeze(0).unsqueeze(0), f"under_image_rec_{i}.png")
        under_image_rec = np.abs(under_image_rec)

        under_image_rec = torch.from_numpy(under_image_rec).to(torch.float32).unsqueeze(0)
        mask = torch.from_numpy(self.mask).to(torch.float32).unsqueeze(0)

        image_rec = np.abs(image_rec)
        image_rec = torch.from_numpy(image_rec).to(torch.float32).unsqueeze(0)

        k = torch.from_numpy(k).to(torch.complex64).unsqueeze(0)

        return (under_image_rec, image_rec, k, mask, torch.from_numpy(self.coil_map))
    
if __name__ == "__main__":
    data_set = Ista_ACDC()
    data_loader = DataLoader(dataset=data_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    for i, sampled_batch in enumerate(tqdm(data_loader)):
        us_image, fs_image, k, us_mask, coil_map = sampled_batch
        print(us_image.shape, fs_image.shape, k.shape, us_mask.shape, coil_map.shape)
        if i == 1:
            break


