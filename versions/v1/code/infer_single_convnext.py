# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from collections import OrderedDict

# 无GUI环境保存图片
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch.utils.data import Dataset

# =========================================================
# 0) Metrics
# =========================================================
def nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / (np.linalg.norm(gt) ** 2 + 1e-12)

def psnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred, data_range=max(gt.max() - gt.min(), 1e-12))

def ssim(gt, pred):
    return structural_similarity(gt, pred, data_range=max(gt.max() - gt.min(), 1e-12))

# =========================================================
# 1) Dataset（按你之前版本整合）
# =========================================================
class SliceData_CC359(Dataset):
    def __init__(self,
                 data_dir="/scratch/pf2m24/data/CCP359/Train/",
                 select='FSPD', type='train', acceleration=8, mask_type='equispaced',
                 resolution=256, rate=1.0):
        self.data_dir = data_dir
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        all_files.sort()
        if rate < 1.0:
            all_files = all_files[: int(len(all_files) * rate)]

        self.examples = []
        for fname in all_files:
            kspace = np.load(fname)  # (S, H, W, 2)
            if kspace.ndim != 4 or kspace.shape[-1] != 2:
                raise ValueError(f"{fname} shape invalid, expected (S,H,W,2), got {kspace.shape}")
            num_slices = kspace.shape[0]
            num_start = 30
            num_end = max(num_slices - 30, num_start + 1)
            self.examples += [(fname, s, None, None) for s in range(num_start, num_end)]

        print("data num:", len(self.examples))

        if mask_type == 'equispaced':
            self.mask = np.load('/scratch/pf2m24/projects/MambaIR/simple_mambair/data_loading/mask_0_256_af8.npy')
            print("mask shape", self.mask.shape)
            print("sum", self.mask[:, 0, :].sum())
        else:
            raise NotImplementedError("Only equispaced mask path provided in this script.")

        self.coil_map = np.ones([1, 256, 256], dtype=np.float32)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_id, _, _ = self.examples[i]
        fully_np = np.load(fname)[slice_id].transpose(2, 0, 1).astype(np.float32)  # [2,H,W]
        fully = fully_np[0] + 1j * fully_np[1]

        # image域
        fully = np.fft.ifftshift(fully)
        image_rec = np.abs(np.fft.ifft2(np.fft.ifftshift(fully)))
        image_rec = self.norm(image_rec)  # [H,W], 0..1

        # 回到 kspace 并掩膜
        fully_k = np.fft.fftshift(np.fft.fft2(image_rec))
        fully_k = np.expand_dims(fully_k, axis=0)  # [1,H,W]

        mask = self.mask
        under_sampling = fully_k * mask  # complex

        # 欠采样 image 域
        under_image_rec = np.fft.ifft2(np.fft.ifftshift(under_sampling))

        # 模型输入为实/虚两通道
        us_img_2ch = np.stack([under_image_rec.real, under_image_rec.imag], axis=0).squeeze()  # [2,H,W]
        # GT 两通道（imag≈0）
        gt_img_2ch = np.stack([image_rec.real, image_rec.imag], axis=0).squeeze()

        us_image = torch.from_numpy(np.ascontiguousarray(us_img_2ch)).to(torch.float32)
        fs_image = torch.from_numpy(np.ascontiguousarray(gt_img_2ch)).to(torch.float32)
        us_mask  = torch.from_numpy(mask.astype(np.float32))
        coil_map = torch.from_numpy(self.coil_map.astype(np.float32))

        sample = dict(
            us_image=us_image,
            fs_image=fs_image,
            us_mask=us_mask,
            coil_map=coil_map,
            fname=fname,
            slice_id=slice_id
        )
        return sample

    @staticmethod
    def norm(image_2D):
        max_ = float(np.max(image_2D))
        min_ = float(np.min(image_2D))
        if max_ - min_ < 1e-12:
            return image_2D
        return (image_2D - min_) / (max_ - min_ + 1e-12)

# =========================================================
# 2) 模型（ConvNeXt encoder + SimpleDecoder）
# =========================================================
class SimpleDecoder(nn.Module):
    def __init__(self, channels=[96, 192, 384, 768], out_channels=2):
        super().__init__()
        self.up3 = self._up_block(channels[3], channels[2])
        self.up2 = self._up_block(channels[2], channels[1])
        self.up1 = self._up_block(channels[1], channels[0])
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(channels[0], 64, kernel_size=4, stride=4),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        x = self.up3(c4)
        x = x + c3
        x = self.up2(x)
        x = x + c2
        x = self.up1(x)
        x = x + c1
        out = self.final_up(x)
        return out

class ConvNextReconNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "convnext_tiny",
            in_chans=2,
            num_classes=2,
            pretrained=False,
            features_only=True
        )
        self.decoder = SimpleDecoder()

    def forward(self, x):
        feats = self.encoder(x)  # 列表 [C1,C2,C3,C4]
        out = self.decoder(feats)
        return out

def _strip_module_prefix(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k[7:]] = v if k.startswith("module.") else v
        if not k.startswith("module."):
            new_sd[k] = v
    return new_sd

def load_model(ckpt_path, device="cuda"):
    model = ConvNextReconNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = _strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        only_bn = all(("running_mean" in k or "running_var" in k or "num_batches_tracked" in k) for k in missing)
        if not only_bn:
            print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)
    model.eval()
    return model

# =========================================================
# 3) 可视化（对比图 + 频谱图）
# =========================================================
def save_compare_figure(us_mag, recon_mag, gt_mag, save_path="recon_compare.png"):
    err = np.abs(recon_mag - gt_mag)
    vmax = max(gt_mag.max(), recon_mag.max(), us_mag.max())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(us_mag, cmap='gray', vmin=0, vmax=vmax);    axes[0].set_title("Under (mag)");  axes[0].axis('off')
    axes[1].imshow(recon_mag, cmap='gray', vmin=0, vmax=vmax); axes[1].set_title("Recon (mag)");  axes[1].axis('off')
    axes[2].imshow(gt_mag, cmap='gray', vmin=0, vmax=vmax);    axes[2].set_title("Target (mag)"); axes[2].axis('off')
    axes[3].imshow(err, cmap='magma');                         axes[3].set_title("Error");        axes[3].axis('off')
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)

# ====== FFT工具 ======
def normalize_feat(x, eps=1e-8):
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    l2 = torch.sqrt((x**2).sum(dim=(-2, -1), keepdim=True) + eps)
    return x / (l2 + eps)

def fft2_magnitude_normed(x, eps=1e-12):
    B, C, H, W = x.shape
    X = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    mag = torch.abs(X) / (H * W)
    return mag.clamp_min(eps)

def shrink_to_14x14(spec):
    t = spec.float()
    H, W = t.shape[-2:]
    ph, pw = max(H // 14, 1), max(W // 14, 1)
    if ph > 1 or pw > 1:
        t = F.avg_pool2d(t, kernel_size=(ph, pw), stride=(ph, pw), ceil_mode=True)
    else:
        t = F.interpolate(t, size=(14, 14), mode='bilinear', align_corners=False)
    return t

def radial_profile_1d(avg_mag_2d):
    H, W = avg_mag_2d.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices((H, W))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    nbins = int(np.ceil(rr.max())) + 1
    prof = np.zeros(nbins, dtype=np.float64)
    cnt = np.zeros(nbins, dtype=np.int64)
    ridx = np.floor(rr).astype(int)
    np.add.at(prof, ridx, avg_mag_2d)
    np.add.at(cnt, ridx, 1)
    prof /= np.maximum(cnt, 1)
    r_norm = np.arange(nbins) / (nbins - 1 + 1e-8)
    return r_norm, prof

def plot_grid_8(spec14_before, spec14_after, save_path="figA_convnext.png"):
    fb = spec14_before.mean(dim=0)  # [C,14,14]
    fa = spec14_after.mean(dim=0)
    C = fb.shape[0]
    idxs = torch.linspace(0, C - 1, 8).long()
    tiles_b, tiles_a = fb[idxs], fa[idxs]
    both = torch.cat([tiles_b.flatten(), tiles_a.flatten()])
    vmax = torch.quantile(both, 0.99).item()
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(tiles_b[i].cpu(), cmap='viridis', vmin=0, vmax=vmax); axes[0, i].axis('off')
        axes[1, i].imshow(tiles_a[i].cpu(), cmap='viridis', vmin=0, vmax=vmax); axes[1, i].axis('off')
    axes[0, 0].set_ylabel("before"); axes[1, 0].set_ylabel("after")
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)

def plot_radial_curve(spec_before, spec_after, save_path="figB_convnext.png", show_delta=False):
    sb = spec_before.mean(dim=(0, 1)).cpu().numpy()
    sa = spec_after.mean(dim=(0, 1)).cpu().numpy()
    r_b, pb = radial_profile_1d(sb)
    r_a, pa = radial_profile_1d(sa)
    yb, ya = np.log(pb + 1e-12), np.log(pa + 1e-12)

    plt.figure(figsize=(5, 4))
    plt.plot(r_b, yb, label='before')
    plt.plot(r_a, ya, label='after')
    if show_delta:
        plt.plot(r_b, (ya - yb), label='Δlog')
    plt.xlabel('Frequency'); plt.ylabel('log amplitude'); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

# =========================================================
# 4) Hook：自动找到 encoder 的“最后一个 conv 主模块”
# =========================================================
def find_last_conv_block_in_encoder(enc):
    """
    自动寻找 encoder 里“最后一个 stage 的最后一个 block”，
    兼容命名：
      - stages_3.blocks.2
      - stages.3.blocks.2
    找不到时回退到 getattr(enc, f"stages_{max}").blocks[-1]
    """
    last = None
    last_tuple = (-1, -1)

    for name, mod in enc.named_modules():
        parts = name.split('.')
        for i, p in enumerate(parts):
            s_idx = None
            # 形式1: 'stages_3'
            if p.startswith('stages_') and p[len('stages_'):].isdigit():
                s_idx = int(p[len('stages_'):])
            # 形式2: 'stages' 后面紧跟数字
            elif p == 'stages' and i + 1 < len(parts) and parts[i + 1].isdigit():
                s_idx = int(parts[i + 1])

            if s_idx is None:
                continue

            # 找 blocks 索引
            b_idx = None
            # 形式 a: ... blocks.2
            if i + 1 < len(parts) and parts[i + 1] == 'blocks':
                if i + 2 < len(parts) and parts[i + 2].isdigit():
                    b_idx = int(parts[i + 2])

            if b_idx is None:
                continue

            t = (s_idx, b_idx)
            if t > last_tuple:
                last_tuple = t
                last = mod

    if last is not None:
        print(f"[Hook] 选用最后模块(遍历命名得到): stage {last_tuple[0]}, block {last_tuple[1]}")
        return last

    # ---------- 回退方案：直接用属性取最高 stage 的最后一个 block ----------
    # 找最大 stage 索引
    max_s = -1
    # 先看 children 名称
    for child_name, _ in enc.named_children():
        if child_name.startswith('stages_'):
            try:
                si = int(child_name.split('_')[1])
                max_s = max(max_s, si)
            except:
                pass
    if max_s >= 0 and hasattr(enc, f"stages_{max_s}"):
        stage = getattr(enc, f"stages_{max_s}")
        if hasattr(stage, 'blocks') and len(stage.blocks) > 0:
            print(f"[Hook] 选用最后模块(属性回退): stage {max_s}, block {len(stage.blocks)-1}")
            return stage.blocks[-1]

    # 再次失败就给出可视化提示
    print("[Error] 仍未定位到目标模块。以下列出包含 'stages' 的模块名：")
    for n, _ in enc.named_modules():
        if 'stages' in n or 'stages_' in n:
            print("  -", n)
    raise RuntimeError("无法自动定位最后一个 conv 主模块，请检查 encoder 结构。")


# =========================================================
# 5) 主流程
# =========================================================
def run_single_infer(ckpt_path,
                     data_dir="/scratch/pf2m24/data/CCP359/Val",
                     index=0,
                     save_dir="./infer_vis_convnext",
                     device=None):
    os.makedirs(save_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 载入模型
    model = load_model(ckpt_path, device=device)

    # 准备数据
    dataset = SliceData_CC359(data_dir=data_dir, type='val', acceleration=8, mask_type='equispaced', resolution=256)
    assert 0 <= index < len(dataset), f"index 越界，0 <= index < {len(dataset)}"
    sample = dataset[index]
    us_image = sample["us_image"].unsqueeze(0).to(device)  # [1,2,H,W]
    fs_image = sample["fs_image"].numpy()                  # [2,H,W]
    fname = sample["fname"]; slice_id = sample["slice_id"]

    # 推理
    with torch.no_grad():
        out = model(us_image)           # [1,2,H,W]
        pred = out[0].cpu().numpy()     # [2,H,W]

    # 幅度图
    us_mag    = np.abs(us_image[0, 0].cpu().numpy() + 1j * us_image[0, 1].cpu().numpy())
    recon_mag = np.abs(pred[0] + 1j * pred[1])
    gt_mag    = np.abs(fs_image[0] + 1j * fs_image[1])

    # 指标
    metric_psnr = psnr(gt_mag, recon_mag)
    metric_ssim = ssim(gt_mag, recon_mag)
    metric_nmse = nmse(gt_mag, recon_mag)
    print(f"[Single] {os.path.basename(fname)} | slice {slice_id} | PSNR={metric_psnr:.4f}  SSIM={metric_ssim:.4f}  NMSE={metric_nmse:.6f}")

    # 保存对比图
    np.save(os.path.join(save_dir, "under.npy"),  us_mag)
    np.save(os.path.join(save_dir, "recon.npy"),  recon_mag)
    np.save(os.path.join(save_dir, "target.npy"), gt_mag)
    save_compare_figure(us_mag, recon_mag, gt_mag, save_path=os.path.join(save_dir, "recon_compare.png"))

    # ============ 频谱分析：hook 在 encoder 的“最后一个 conv 主模块” ============
    enc = model.encoder
    target_block = find_last_conv_block_in_encoder(enc)

    cache = {"before": None, "after": None}
    def _pre_hook(m, inputs):
        x = inputs[0]
        if torch.is_tensor(x) and x.ndim == 4:
            cache["before"] = x.detach()
    def _post_hook(m, inputs, out):
        y = out
        if torch.is_tensor(y) and y.ndim == 4:
            cache["after"] = y.detach()

    h1 = target_block.register_forward_pre_hook(_pre_hook)
    h2 = target_block.register_forward_hook(_post_hook)

    # 再跑一次前向，用于抓特征
    with torch.no_grad():
        _ = model(us_image)

    h1.remove(); h2.remove()

    if cache["before"] is None or cache["after"] is None:
        print("[FFT] 未抓到 BCHW 特征，请检查模块选择。")
    else:
        feat_b = normalize_feat(cache["before"])
        feat_a = normalize_feat(cache["after"])
        mag_b  = fft2_magnitude_normed(feat_b)
        mag_a  = fft2_magnitude_normed(feat_a)

        spec14_b = shrink_to_14x14(mag_b)
        spec14_a = shrink_to_14x14(mag_a)
        plot_grid_8(spec14_b, spec14_a, save_path=os.path.join(save_dir, "figA_convnext.png"))
        plot_radial_curve(mag_b, mag_a, save_path=os.path.join(save_dir, "figB_convnext.png"), show_delta=False)
        print("[FFT] Saved:",
              os.path.join(save_dir, "figA_convnext.png"),
              os.path.join(save_dir, "figB_convnext.png"))

    return dict(psnr=metric_psnr, ssim=metric_ssim, nmse=metric_nmse,
                save_dir=os.path.abspath(save_dir),
                fname=fname, slice_id=int(slice_id))

# =========================================================
# 6) CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=False,
                        default="/scratch/pf2m24/projects/MRIRecon/MambaRecon/model/mamba_unrolled_140_Patch_2_convnext_4x_1111/mamba_unrolled/mamba_unrolled_best_psnr_model.pth")
    parser.add_argument("--data_dir", type=str, default="/scratch/pf2m24/data/CCP359/Val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="/scratch/pf2m24/projects/MRIRecon/MambaRecon/inference_results/convnext")
    args = parser.parse_args()

    res = run_single_infer(args.ckpt, data_dir=args.data_dir, index=args.index, save_dir=args.save_dir)
    print("Done. Results saved to:", res["save_dir"])
