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
from dataloaders.ACDC_dataset_r32 import Ista_ACDC
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from networks.ista_net import ISTANetPlus

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
        from networks.vision_mamba_ACDC import MambaUnrolled as VIM_seg
    elif args.model == "mamba_unet":
        from networks.vision_mamba_ACDC import MambaUnet as VIM_seg
    elif args.model == "swin_unet":
        from networks.vision_transformer import SwinUnet as VIM_seg
        args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
    elif args.model == "swin_unrolled":
        from networks.vision_transformer import SwinUnrolled as VIM_seg
        args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"

    config = get_config(args)
    
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
    train_dataset = Ista_ACDC(
        data_dir="/scratch/pf2m24/data/ACDC/training",
        mask_path="/scratch/pf2m24/projects/MRIRecon/MambaReconV3/code/dataloaders/radial_mask/radial_mask_16.npy"
    )
    
    val_dataset = Ista_ACDC(
        data_dir="/scratch/pf2m24/data/ACDC/testing",
        mask_path="/scratch/pf2m24/projects/MRIRecon/MambaReconV3/code/dataloaders/radial_mask/radial_mask_16.npy"
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

    # model = VIM_seg(config, patch_size=args.patch_size, num_classes=2).cuda(local_rank)
    model = ISTANetPlus().cuda(local_rank)
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
                
    
    for epoch in range(start_epoch, max_epoch + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        pbar = tqdm(trainloader, ncols=120, desc=f'[Train] Epoch {epoch}/{max_epoch}', unit='batch', leave=False, disable=not is_main)

        for i_batch, sampled_batch in enumerate(pbar):
            inputs, target, masked_kspace, mask, fname = sampled_batch

            #print(inputs.shape, target.shape, masked_kspace.shape, mask.shape)
            inputs = inputs.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)
            masked_kspace = masked_kspace.cuda(local_rank, non_blocking=True)
            mask = mask.cuda(local_rank, non_blocking=True)
            fname = fname

            img_out, loss_layers_sym = model(inputs, masked_kspace, mask)
            loss_discrepancy = torch.mean(torch.pow(img_out - target, 2))
            loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
            for k in range(8-1):
                loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))
            gamma = torch.Tensor([0.01]).cuda(local_rank, non_blocking=True)
            loss = loss_discrepancy + torch.mul(gamma, loss_constraint)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = adjust_learning_rate(optimizer, epoch)

            iter_num += 1
            if is_main and writer:
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss.item(), iter_num)
                pbar.set_postfix({'loss': loss.item(), 'lr': lr_})

        # 验证
        model.eval()
        total_psnr = total_rmse = total_ssim = 0.0
        tbar = tqdm(valloader, ncols=100, desc=f'[Val] Epoch {epoch}', disable=not is_main)
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(tbar):
                inputs, target, masked_kspace, mask, fname = sampled_batch
                inputs = inputs.cuda(local_rank, non_blocking=True)
                target = target.cuda(local_rank, non_blocking=True)
                masked_kspace = masked_kspace.cuda(local_rank, non_blocking=True)
                mask = mask.cuda(local_rank, non_blocking=True)
                fname = fname

                img_out, loss_layers_sym = model(inputs, masked_kspace, mask)
                loss_discrepancy = torch.mean(torch.pow(img_out - target, 2))
                loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
                for k in range(8-1):
                    loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))
                gamma = torch.Tensor([0.01]).cuda(local_rank, non_blocking=True)
                loss = loss_discrepancy + torch.mul(gamma, loss_constraint)

                total_psnr += peak_signal_noise_ratio(target.cpu().numpy(), img_out.clamp(0, 1).cpu().numpy(), data_range=target.max())
                total_rmse += nmse(target.cpu().numpy(), img_out.clamp(0, 1).cpu().numpy())
                total_ssim += calculate_ssim(target.cpu().numpy(), img_out.clamp(0, 1).cpu().numpy())
        # 聚合所有进程的结果（取平均）
        n_val = len(valloader)
        total_psnr = torch.tensor(total_psnr / n_val, device=local_rank)
        total_ssim = torch.tensor(total_ssim / n_val, device=local_rank)
        total_rmse = torch.tensor(total_rmse / n_val, device=local_rank)

        # All-reduce（多卡结果同步取平均）
        dist.all_reduce(total_psnr, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_ssim, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_rmse, op=dist.ReduceOp.SUM)
        total_psnr /= world_size
        total_ssim /= world_size
        total_rmse /= world_size

        if is_main and writer:
            writer.add_scalar('info/val_mean_psnr', total_psnr.item(), iter_num)
            writer.add_scalar('info/val_mean_ssim', total_ssim.item(), iter_num)
            writer.add_scalar('info/val_mean_rmse', total_rmse.item(), iter_num)

        if is_main:
            # 保存 PSNR 最优模型
            if total_psnr > best_psnr:
                best_psnr = total_psnr
                best_psnr_epoch = epoch
                checkpoint_psnr = {
                    'epoch': epoch,
                    'iter_num': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    'best_ssim': best_ssim
                }
                # torch.save(checkpoint_psnr, os.path.join(snapshot_path, f'iter_{iter_num}_best_psnr.pth'))
                torch.save(checkpoint_psnr, os.path.join(snapshot_path, f'{args.model}_best_psnr_model.pth'))

            # 保存 SSIM 最优模型
            if total_ssim > best_ssim:
                best_ssim = total_ssim
                best_ssim_epoch = epoch
                checkpoint_ssim = {
                    'epoch': epoch,
                    'iter_num': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    'best_ssim': best_ssim
                }
                # torch.save(checkpoint_ssim, os.path.join(snapshot_path, f'iter_{iter_num}_best_ssim.pth'))
                torch.save(checkpoint_ssim, os.path.join(snapshot_path, f'{args.model}_best_ssim_model.pth'))
            if total_rmse < best_rmse:
                best_rmse = total_rmse
                best_rmse_epoch = epoch


            checkpoint_lateset = {
                    'epoch': epoch,
                    'iter_num': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    'best_ssim': best_ssim
                }
            torch.save(checkpoint_lateset, os.path.join(snapshot_path, f'{args.model}_latest_model.pth'))

            # 日志记录
            logging.info(f'[Epoch {epoch}] Val PSNR: {total_psnr:.4f}, SSIM: {total_ssim:.4f}, RMSE: {total_rmse:.4f}')
            logging.info(f'Best PSNR so far (epoch {best_psnr_epoch}): {best_psnr:.4f}')
            logging.info(f'Best SSIM so far (epoch {best_ssim_epoch}): {best_ssim:.4f}')
            logging.info(f'Best RMSE so far (epoch {best_rmse_epoch}): {best_rmse:.4f}')

        model.train()

    if is_main and writer:
        writer.close()

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
    parser.add_argument('--resume', default=None)
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
    parser.add_argument('--name', type=str, default='Ista_ACDC_16')
    args = parser.parse_args()
    main_worker(args)
