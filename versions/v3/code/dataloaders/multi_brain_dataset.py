from importlib.metadata import files
import glob, os, h5py

import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np

from torchvision.utils import save_image


class SliceData_multi(Dataset):
    def __init__(self, 
                data_dir="/mnt/afs/fpc_projects/data/M4Raw/multicoil_train", mask_path="/mnt/afs/fpc_projects/MambaRecon/code/dataloaders/multi_mask/mask_0_256_af8.npy"):
        hits = glob.glob(os.path.join(data_dir, "**", "*T1*.h5"), recursive=True)
        self.mask = np.load(mask_path)
        self.files = []
        for i in range(18):
            for h in hits:
                self.files.append((h, i))
        print("self.mask.shape:", self.mask.shape)
        self.coil_map = np.ones([4,256,256], dtype=np.float32)
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname, slice_num = self.files[i]
        with h5py.File(fname, 'r') as f:
            k = f["kspace"][:][slice_num]
            rss_image = f["reconstruction_rss"][:][slice_num] 
        k = k * self.mask
        # max_val = np.max(np.abs(rss_image))
        rss_image = (rss_image - np.min(rss_image)) / (np.max(rss_image) - np.min(rss_image))
        under_rss = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(k)))
        # print("under_rss shape before:", under_rss.shape)
        # under_rss =np.sqrt(np.sum(np.abs(under_rss) ** 2, axis=0, keepdims=True))
        under_rss = np.concatenate([under_rss.real, under_rss.imag], axis=0).squeeze()
        rss_image = np.stack([rss_image.real, rss_image.imag], axis=0).squeeze()

        rss_image = torch.from_numpy(rss_image).to(torch.float32)
        under_rss = torch.from_numpy(under_rss).to(torch.float32)

        return dict(us_image=under_rss, 
                    fs_image=rss_image, 
                    us_mask=torch.from_numpy(self.mask).to(torch.float32), 
                    coil_map=torch.from_numpy(self.coil_map).to(torch.float32))