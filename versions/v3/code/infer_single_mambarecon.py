import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# ====== 关键：与训练一致的导入 ======
from config import get_config
from dataloaders.CC359_dataset_PGIUN_8 import SliceData_CC359

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 根据 --model 选择具体类（与训练完全一致）
def build_model(args, device):
    if args.model == "mamba_unrolled":
        from networks.vision_mamba import MambaUnrolled as VIM_seg
    elif args.model == "mamba_unet":
        from networks.vision_mamba import MambaUnet as VIM_seg
    elif args.model == "swin_unet":
        from networks.vision_transformer import SwinUnet as VIM_seg
        # 训练里给过默认 cfg；这里保持一致（若命令行未显式传入）
        if args.cfg is None:
            args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
    elif args.model == "swin_unrolled":
        from networks.vision_transformer import SwinUnrolled as VIM_seg
        if args.cfg is None:
            args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    config = get_config(args)
    model = VIM_seg(config, patch_size=args.patch_size, num_classes=2).to(device)
    model.eval()
    return model

# ============== metrics ==============
def nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / (np.linalg.norm(gt) ** 2 + 1e-12)

def psnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred, data_range=max(gt.max() - gt.min(), 1e-12))

def ssim(gt, pred):
    return structural_similarity(gt, pred, data_range=max(gt.max() - gt.min(), 1e-12))

# ============== ckpt 处理（去 module. 前缀） ==============
def _strip_module_prefix(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    return new_sd

def load_weights(model, ckpt_path, device):
    raw = torch.load(ckpt_path, map_location=device)
    sd = raw.get("model_state_dict", raw.get("state_dict", raw))
    if not isinstance(sd, dict):
        raise RuntimeError("Invalid checkpoint format, no state dict found.")
    sd = _strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    # 仅忽略 BN 统计等非关键 buffer
    if missing:
        only_bn = all(("running_mean" in k or "running_var" in k or "num_batches_tracked" in k) for k in missing)
        if not only_bn:
            print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)

# ============== 可视化保存 ==============
def save_compare_figure(us_mag, recon_mag, gt_mag, save_path):
    err = np.abs(recon_mag - gt_mag)
    vmax = max(us_mag.max(), recon_mag.max(), gt_mag.max())
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(us_mag, cmap='gray', vmin=0, vmax=vmax);    axes[0].set_title("Under (mag)");  axes[0].axis('off')
    axes[1].imshow(recon_mag, cmap='gray', vmin=0, vmax=vmax); axes[1].set_title("Recon (mag)");  axes[1].axis('off')
    axes[2].imshow(gt_mag, cmap='gray', vmin=0, vmax=vmax);    axes[2].set_title("Target (mag)"); axes[2].axis('off')
    axes[3].imshow(err, cmap='magma');                         axes[3].set_title("Error");        axes[3].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def tensor_to_mag(x):
    """
    接受 torch.Tensor 或 numpy.ndarray，返回 2D numpy 幅度图 [H,W]
    兼容形状：
      [B,2,H,W] -> sqrt(real^2+imag^2) 取第一个batch
      [B,1,H,W] -> 取第一个batch的单通道
      [2,H,W]   -> 实虚
      [1,H,W]   -> 单通道
      [H,W]     -> 已是2D
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    # 去 batch 维（若存在且为1）
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]

    if x.ndim == 3:
        C, H, W = x.shape
        if C == 2:
            mag = torch.sqrt(x[0]**2 + x[1]**2)
            return mag.cpu().numpy()
        elif C == 1:
            return x[0].cpu().numpy()
        else:
            raise ValueError(f"Unexpected channels: {C}, expect 1 or 2.")
    elif x.ndim == 2:
        return x.cpu().numpy()
    else:
        raise ValueError(f"Unexpected shape: {tuple(x.shape)}")
    
# ============== 主流程 ==============
def main():
    parser = argparse.ArgumentParser()
    # —— 与训练保持一致的关键参数 ——
    parser.add_argument('--model', type=str, default='mamba_unrolled',
                        choices=['mamba_unrolled', 'mamba_unet', 'swin_unet', 'swin_unrolled'])
    parser.add_argument('--cfg', type=str, default='../code/configs/vmamba_tiny.yaml')
    parser.add_argument('--patch_size', type=int, default=2)

    # —— 推理相关 ——
    # /scratch/pf2m24/projects/MRIRecon/MambaReconV3/model/mamba_unrolled_140_Patch_2_cc359_p2_8x2_moe_fpc_8x/mamba_unrolled/mamba_unrolled_best_psnr_model.pth
    # /scratch/pf2m24/projects/MRIRecon/MambaReconV3/model/mamba_unrolled_140_Patch_2_cc359_p2_8x2_moe_dim96_8x/mamba_unrolled/mamba_unrolled_best_psnr_model.pth
    parser.add_argument('--ckpt', type=str, default='/scratch/pf2m24/projects/MRIRecon/MambaReconV3/model/mamba_unrolled_140_Patch_2_cc359_p2_8x2_moe_dim96_8x/mamba_unrolled/mamba_unrolled_best_psnr_model.pth')
    parser.add_argument('--data_dir', type=str, default='/scratch/pf2m24/data/CCP359/Val',
                        help='folder with .npy volumes for CC359')
    parser.add_argument('--index', type=int, default=0,
                        help='dataset index (按构造顺序第 index 个样本)')
    parser.add_argument('--save_dir', type=str, default='/scratch/pf2m24/projects/MRIRecon/MambaRecon/inference_results/hifimamba',
                        help='where to save outputs')
    args = parser.parse_args()

    # 补齐 config.update_config 会访问到的字段，避免 AttributeError
    defaults = {
        'opts': None,
        'batch_size': None,
        'zip': False,
        'cache_mode': None,
        'resume': '',
        'accumulation_steps': 0,
        'use_checkpoint': False,
        'amp_opt_level': '',
        'tag': None,
        'eval': False,
        'throughput': False,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)


    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) 构建与训练一致的模型并加载权重
    model = build_model(args, device)
    from thop import profile, clever_format
    edge = 320
    input_shape = (2, edge, edge)
    input_tensor = torch.randn(1, *input_shape).to(device)
    mask = torch.randn(1, 1, edge, edge).to(device)
    coil_map = torch.randn(1, 1, edge, edge).to(device)
    flops, params = profile(model, inputs=(input_tensor, mask, coil_map), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" % (flops))
    print("params: %s" % (params))
    # load_weights(model, args.ckpt, device)

    # 2) 准备数据（直接复用你训练用的 Dataset）
    dataset = SliceData_CC359(
        data_dir=args.data_dir,
        acceleration=8,
        mask_type='equispaced',
        resolution=256,
        type='val',
    )
    assert 0 <= args.index < len(dataset), f"index 越界：0 <= index < {len(dataset)}"
    sample = dataset[args.index]

    us_image = sample['us_image'].unsqueeze(0).to(device)   # [1,2,H,W]
    fs_image = sample['fs_image'].numpy()                   # [2,H,W] (numpy for metrics)
    us_mask  = sample['us_mask'].unsqueeze(0).to(device)    # [1,1,H,W] or [1,H,W] 依据你的实现
    coil_map = sample['coil_map'].unsqueeze(0).to(device)   # [1,1,H,W]

    # 3) 前向推理（与训练一致的调用签名）
    # with torch.no_grad():
    #     pred = model(us_image, us_mask, coil_map)           # [1,2,H,W]
    #     pred = pred[0].detach().cpu().numpy()               # [2,H,W]
    import time
    from tqdm import tqdm
    model.eval()
    torch.backends.cudnn.benchmark = True  # 允许cudnn选择最快算法（固定尺寸下更快）

    WARMUP = 40
    RUNS = 100

    with torch.no_grad():
        # 预热
        for _ in tqdm(range(WARMUP)):
            _ = model(us_image, us_mask, coil_map)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 计时
        total_ms = 0.0
        last_out = None
        if torch.cuda.is_available():
            for _ in tqdm(range(RUNS)):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                last_out = model(us_image, us_mask, coil_map)
                end.record()
                torch.cuda.synchronize()
                total_ms += start.elapsed_time(end)  # 毫秒

        avg_ms = total_ms / RUNS
        fps = 1000.0 / avg_ms if avg_ms > 0 else float('inf')
        print(f"[Speed] avg latency per image: {avg_ms:.3f} ms | FPS: {fps:.2f}")

        # 取最后一次输出做后续处理/可视化
        pred = last_out[0].detach().cpu().numpy()   # [2,H,W]


    print("=== Debug Shapes ===")
    print('pred shape:', pred.shape)
    # 4) 复数 → 幅度
    # us_mag    = np.abs(us_image[0,0].detach().cpu().numpy() + 1j * us_image[0,1].detach().cpu().numpy())
    # recon_mag = np.abs(pred[0] + 1j * pred[1])
    # gt_mag    = np.abs(fs_image[0] + 1j * fs_image[1])
    # 不要先写 pred = out[0] 再索引；直接交给上面的函数处理
    recon_mag = tensor_to_mag(pred)          # 预测幅度
    us_mag    = tensor_to_mag(us_image)     # 欠采样幅度
    gt_mag    = tensor_to_mag(fs_image)                    # GT（dataset返回的 numpy 也可直接传）

    # 5) 指标
    metric_psnr = psnr(gt_mag, recon_mag)
    metric_ssim = ssim(gt_mag, recon_mag)
    metric_nmse = nmse(gt_mag, recon_mag)
    print(f"[Single] PSNR={metric_psnr:.4f}  SSIM={metric_ssim:.4f}  NMSE={metric_nmse:.6f}")

    # 6) 保存结果
    np.save(os.path.join(args.save_dir, "under.npy"),  us_mag)
    np.save(os.path.join(args.save_dir, "recon.npy"),  recon_mag)
    np.save(os.path.join(args.save_dir, "target.npy"), gt_mag)

    save_compare_figure(us_mag, recon_mag, gt_mag,
                        save_path=os.path.join(args.save_dir, "recon_compare.png"))

    # 同时各自保存单图（方便论文挑图）
    plt.imsave(os.path.join(args.save_dir, "under.png"),  us_mag,    cmap='gray', vmin=0, vmax=max(us_mag.max(), 1.0))
    plt.imsave(os.path.join(args.save_dir, "recon.png"),  recon_mag, cmap='gray', vmin=0, vmax=max(recon_mag.max(), 1.0))
    plt.imsave(os.path.join(args.save_dir, "target.png"), gt_mag,    cmap='gray', vmin=0, vmax=max(gt_mag.max(), 1.0))

    print("Saved to:", os.path.abspath(args.save_dir))

if __name__ == "__main__":
    main()
