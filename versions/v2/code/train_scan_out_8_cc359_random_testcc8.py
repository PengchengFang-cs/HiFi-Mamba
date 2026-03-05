import os
import sys
import argparse
import random
import shutil
import logging
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from tensorboardX import SummaryWriter

from config import get_config
from dataloaders.CC359_dataset_PGIUN_random_8 import SliceData_CC359
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

def adjust_learning_rate(optimizer, epoch, total_epochs=72, lr=8e-4, warmup_epochs=5, min_lr=1e-6):
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def calculate_ssim(img1, img2):
    ssim_values = []
    batch_size, channels, width, height = img1.shape
    for i in range(batch_size):
        for j in range(channels):
            ssim_value = structural_similarity(img1[i, j], img2[i, j], channel_axis=None, data_range=img1[i, j].max())
            ssim_values.append(ssim_value)
    return np.mean(ssim_values)

def main_worker(args):
    local_rank, rank, world_size = setup_ddp()
    is_main = (rank == 0)

    if args.model == "mamba_unrolled":
        from networks.vision_mamba import MambaUnrolled as VIM_seg
    elif args.model == "mamba_unet":
        from networks.vision_mamba import MambaUnet as VIM_seg
    elif args.model == "swin_unet":
        from networks.vision_transformer import SwinUnet as VIM_seg
        args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
    elif args.model == "swin_unrolled":
        from networks.vision_transformer import SwinUnrolled as VIM_seg
        args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"

    config = get_config(args)
    

    # 保证每个进程随机种子不同
    if args.seed != -1:
        print('固定种子')
        cudnn.benchmark = not args.deterministic
        cudnn.deterministic = args.deterministic
        seed = args.seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.set_device(local_rank)
        # torch.cuda.manual_seed_all(seed)
    else:
        print('随机种子')

    # 路径和日志
    snapshot_path = f"../model/{args.exp}_{args.labeled_num}_Patch_{args.patch_size}_{args.name}/{args.model}"
    if is_main:
        os.makedirs(snapshot_path, exist_ok=True)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
        logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        writer = SummaryWriter(snapshot_path + '/log')
    else:
        writer = None

    # 数据集与采样器
    train_dataset = SliceData_CC359(
        data_dir="/scratch/pf2m24/data/CCP359/Train",
        acceleration=8,
        mask_type='equispaced',
        resolution=256,
        type="train_",
    )
    val_dataset = SliceData_CC359(
        data_dir="/scratch/pf2m24/data/CCP359/Val",
        acceleration=8,
        mask_type='equispaced',
        resolution=256,
        type="val",
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    trainloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,  # DDP用sampler控制
                            sampler=train_sampler,
                            num_workers=8,
                            pin_memory=True,
                            prefetch_factor=4,
                            drop_last=True)
    valloader = DataLoader(val_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        sampler=val_sampler,
                        num_workers=8,
                        pin_memory=True,
                        prefetch_factor=4,
                        drop_last=False)

    model = VIM_seg(config, patch_size=args.patch_size, num_classes=2).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr)

    start_epoch = 1
    iter_num = 0
    max_epoch = 100
    best_psnr = -1
    best_ssim = -1
    best_rmse = 100
    best_psnr_epoch = 0
    best_ssim_epoch = 0
    best_rmse_epoch = 0

    # Resume training if resume path is provided
    if args.resume is not None and os.path.isfile(args.resume):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.resume, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        # Manually set starting epoch and iter number if known
        # start_epoch = 13  # Manually set, since you've trained 32 epochs
        # iter_num = (start_epoch - 1) * len(trainloader)  # Recover iter_num approx

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        iter_num = checkpoint.get('iter_num', 0)
        best_psnr = checkpoint.get('best_psnr', 0)
        best_ssim = checkpoint.get('best_ssim', 0)
        
        logging.info(f"Successfully resumed from {args.resume}, continuing from epoch {start_epoch}")
    elif args.resume is not None:
        logging.warning(f"Resume checkpoint not found at {args.resume}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.stride() != param.contiguous().stride():
                print(f"[Warning] Param '{name}' has non-contiguous strides: {param.shape}, {param.stride()}")
                
    
    for epoch in range(start_epoch, start_epoch + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        
        model.eval()
        total_psnr = 0.0
        total_rmse = 0.0
        total_ssim = 0.0
        tbar = tqdm(valloader, ncols=100, desc=f'[Val] Epoch {epoch}', disable=not is_main)
        image_path = 'image_cc8'
        from pathlib import Path
        from torchvision.utils import save_image
        image_path = Path(image_path)
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(tbar):
                us_image = sampled_batch['us_image'].cuda(local_rank, non_blocking=True)
                fs_target = sampled_batch['fs_image'].cuda(local_rank, non_blocking=True)
                us_mask = sampled_batch['us_mask'].cuda(local_rank, non_blocking=True)
                coil_maps = sampled_batch['coil_map'].cuda(local_rank, non_blocking=True)

                outputs = model(us_image, us_mask, coil_maps)
                outputs = torch.abs(outputs[:, 0, :, :] + 1j * outputs[:, 1, :, :]).unsqueeze(1)
                fs_target_abs = torch.abs(fs_target[:, 0, :, :] + 1j * fs_target[:, 1, :, :]).unsqueeze(1)

                o_np = outputs.clamp(0, 1).cpu().numpy()
                t_np = fs_target_abs.clamp(0, 1).cpu().numpy()

                total_psnr += peak_signal_noise_ratio(t_np, o_np, data_range=t_np.max())
                total_rmse += nmse(t_np, o_np)
                total_ssim += calculate_ssim(t_np, o_np)
                if i_batch > 100 and i_batch <150 and image_path is not None:
                
                    if not image_path.exists():
                        image_path.mkdir(parents=True, exist_ok=True)

                    # 保存重建图像
                    save_image(outputs, image_path / f'output_idx_{i_batch}.png', normalize=True)

        # ---- reduce across ranks ----
        n_val = len(valloader)
        total_psnr = torch.tensor(total_psnr / n_val, device=local_rank)
        total_ssim = torch.tensor(total_ssim / n_val, device=local_rank)
        total_rmse = torch.tensor(total_rmse / n_val, device=local_rank)

        dist.all_reduce(total_psnr, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_ssim, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_rmse, op=dist.ReduceOp.SUM)
        total_psnr /= world_size
        total_ssim /= world_size
        total_rmse /= world_size

        # if is_main and writer:
        #     writer.add_scalar('info/val_mean_psnr', total_psnr.item(), iter_num)
        #     writer.add_scalar('info/val_mean_ssim', total_ssim.item(), iter_num)
        #     writer.add_scalar('info/val_mean_rmse', total_rmse.item(), iter_num)

        if is_main:
            # ---- save best by PSNR ----
            if total_psnr > best_psnr:
                best_psnr = total_psnr
                best_psnr_epoch = epoch
                checkpoint_psnr = {
                    'epoch': epoch,
                    'iter_num': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  # 保存 scheduler
                    'best_psnr': best_psnr,
                    'best_ssim': best_ssim,
                }
                # torch.save(checkpoint_psnr, os.path.join(snapshot_path, f'{args.model}_best_psnr_model.pth'))

            # ---- save best by SSIM ----
            if total_ssim > best_ssim:
                best_ssim = total_ssim
                best_ssim_epoch = epoch
                checkpoint_ssim = {
                    'epoch': epoch,
                    'iter_num': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  # 保存 scheduler
                    'best_psnr': best_psnr,
                    'best_ssim': best_ssim,
                }
                # torch.save(checkpoint_ssim, os.path.join(snapshot_path, f'{args.model}_best_ssim_model.pth'))

                # if epoch == 6:
                #     torch.save(checkpoint_ssim, os.path.join(snapshot_path, f'{args.model}_test_model.pth'))

            # if total_rmse < best_rmse:
            #     best_rmse = total_rmse
            #     best_rmse_epoch = epoch

            # checkpoint_latest = {
            #     'epoch': epoch,
            #     'iter_num': iter_num,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'scheduler_state_dict': scheduler.state_dict(),  # 保存 scheduler
            #     'best_psnr': best_psnr,
            #     'best_ssim': best_ssim,
            # }
            # # torch.save(checkpoint_latest, os.path.join(snapshot_path, f'{args.model}_latest_model.pth'))

            logging.info(f'[Epoch {epoch}] Val PSNR: {total_psnr:.4f}, SSIM: {total_ssim:.4f}, RMSE: {total_rmse:.4f}')
            logging.info(f'Best PSNR so far (epoch {best_psnr_epoch}): {best_psnr:.4f}')
            logging.info(f'Best SSIM so far (epoch {best_ssim_epoch}): {best_ssim:.4f}')
            logging.info(f'Best RMSE so far (epoch {best_rmse_epoch}): {best_rmse:.4f}')

        model.train()


    dist.destroy_process_group()
    return "Training Finished!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ...（保持你的参数不变，略去参数部分，直接复制你的参数设置）
    parser.add_argument('--root_path', type=str, default='../data/ACDC')
    parser.add_argument('--exp', type=str, default='mamba_unrolled')
    parser.add_argument('--dataset', type=str, default='ixi')
    parser.add_argument('--model', type=str, default='mamba_unrolled')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--cfg', type=str, default="../code/configs/vmamba_tiny.yaml")
    parser.add_argument('--opts', default=None, nargs='+')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'])
    parser.add_argument('--resume', default='/scratch/pf2m24/projects/MRIRecon/MambaReconV2/model/mamba_unrolled_140_Patch_1_cc359_p1_8x2_final_random_4x_dw_5_0903/mamba_unrolled/mamba_unrolled_best_psnr_model.pth')
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'])
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--max_iterations', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--deterministic', type=bool, default=True)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--labeled_num', type=int, default=140)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--name', type=str, default='cc359_8x_able_8x2')
    args = parser.parse_args()
    main_worker(args)
