import os
import sys
import argparse
import random
import shutil
import logging
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
from dataloaders.fastMRI_dataset_PGIUN import fastMRI_dataset
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def calculate_ssim(img1, img2):
    ssim_values = []
    batch_size, channels, width, height = img1.shape
    for i in range(batch_size):
        for j in range(channels):
            ssim_value = structural_similarity(
                img1[i, j], img2[i, j], channel_axis=None, data_range=img1[i, j].max()
            )
            ssim_values.append(ssim_value)
    return np.mean(ssim_values)


def build_scheduler(optimizer, steps_per_epoch, max_epoch, warmup_epochs, eta_min=1e-6, iter_num=0):
    """
    构建按-iter的  warmup -> cosine  调度器，并在需要时将其“跳转”到 iter_num 的位置，
    以避免 resume 时再经历一次 warmup。
    """
    total_steps = max_epoch * steps_per_epoch
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    warmup = LambdaLR(optimizer, lr_lambda=lambda s: min(1.0, (s + 1) / max(1, warmup_steps)))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # 关键：直接跳到 iter_num（不重放 warmup）
    if iter_num > 0:
        scheduler.step(iter_num)

    return scheduler


def main_worker(args):
    local_rank, rank, world_size = setup_ddp()
    is_main = (rank == 0)

    # ---- model select ----
    if args.model == "mamba_unrolled":
        from networks.vision_mamba import MambaUnrolled as VIM_seg
        print("mamba_unrolled")
    elif args.model == "mamba_unet":
        from networks.vision_mamba import MambaUnet as VIM_seg
        print("mamba_unet")
    elif args.model == "swin_unet":
        from networks.vision_transformer import SwinUnet as VIM_seg
        args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
    elif args.model == "swin_unrolled":
        from networks.vision_transformer import SwinUnrolled as VIM_seg
        args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    config = get_config(args)

    # ---- seeds ----
    if args.seed != -1:
        if is_main:
            print("固定种子")
        cudnn.benchmark = not args.deterministic
        cudnn.deterministic = args.deterministic
        seed = args.seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.set_device(local_rank)
    else:
        if is_main:
            print("随机种子")


    # # ---- paths & logging ----
    # snapshot_path = f"../model/{args.exp}_{args.labeled_num}_Patch_{args.patch_size}_{args.name}/{args.model}"
    # if is_main:
    #     os.makedirs(snapshot_path, exist_ok=True)
    #     code_dst = os.path.join(snapshot_path, "code")
    #     if os.path.exists(code_dst):
    #         shutil.rmtree(code_dst)
    #     shutil.copytree(".", code_dst, shutil.ignore_patterns([".git", "__pycache__"]))
    #     logging.basicConfig(
    #         filename=os.path.join(snapshot_path, "log.txt"),
    #         level=logging.INFO,
    #         format='[%(asctime)s.%(msecs)03d] %(message)s',
    #         datefmt='%H:%M:%S',
    #     )
    #     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    #     logging.info(str(args))
    #     writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    # else:
    #     writer = None

    # ---- datasets & samplers ----
    train_dataset = fastMRI_dataset(
        data_dir="/scratch/pf2m24/data/FastMRI/singlecoil_train",
        acceleration=8,
        mask_type='equispaced',
        resolution=320,
        type="train_",
    )
    val_dataset = fastMRI_dataset(
        data_dir="/scratch/pf2m24/data/FastMRI/singlecoil_val",
        acceleration=8,
        mask_type='equispaced',
        resolution=320,
        type="val",
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    print('训练集样本数：', len(train_dataset))
    print('验证集样本数：', len(val_dataset))
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # DDP 用 sampler
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=False,
    )

    # ---- train states ----
    start_epoch = 1
    iter_num = 0
    max_epoch = 100
    best_psnr = -1
    best_ssim = -1
    best_rmse = 100
    best_psnr_epoch = 0
    best_ssim_epoch = 0
    best_rmse_epoch = 0

    # ---- model & optimizer ----
    model = VIM_seg(config, patch_size=args.patch_size, num_classes=2).cuda(local_rank)

    from thop import profile, clever_format
    input_shape = (2, 320, 320)
    input_tensor = torch.randn(1, *input_shape).cuda(local_rank)
    mask = torch.randn(1, 1, 320, 320).cuda(local_rank)
    coil_map = torch.randn(1, 1, 320, 320).cuda(local_rank)
    flops, params = profile(model, inputs=(input_tensor, mask, coil_map), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" % (flops))
    print("params: %s" % (params))
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr)

    # ---- scheduler (先构造，后根据 ckpt 决定跳转/加载) ----
    steps_per_epoch = len(trainloader)
    scheduler = build_scheduler(
        optimizer=optimizer,
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
        warmup_epochs=args.warmup_epochs,
        eta_min=1e-6,
        iter_num=0,  # 暂时 0，占个位，下面 resume 再对齐
    )

    # ---- resume (优先恢复 scheduler 的 state_dict；否则用 step(iter_num) 对齐) ----
    if args.resume is not None and os.path.isfile(args.resume):
        map_location = {f'cuda:{0}': f'cuda:{local_rank}'}
        checkpoint = torch.load(args.resume, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        iter_num = checkpoint.get('iter_num', 0)
        best_psnr = checkpoint.get('best_psnr', -1)
        best_ssim = checkpoint.get('best_ssim', -1)

        # 1) 有就直接加载 scheduler 的状态 —— 最稳，不会 re-warmup
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                if is_main:
                    logging.warning(f"Load scheduler_state_dict failed ({e}), fallback to step(iter_num).")
                scheduler = build_scheduler(
                    optimizer=optimizer,
                    steps_per_epoch=steps_per_epoch,
                    max_epoch=max_epoch,
                    warmup_epochs=args.warmup_epochs,
                    eta_min=1e-6,
                    iter_num=iter_num,
                )
        else:
            # 2) 老 ckpt 兼容：没有 scheduler，就跳转到 iter_num
            scheduler = build_scheduler(
                optimizer=optimizer,
                steps_per_epoch=steps_per_epoch,
                max_epoch=max_epoch,
                warmup_epochs=args.warmup_epochs,
                eta_min=1e-6,
                iter_num=iter_num,
            )

        if is_main:
            logging.info(f"Successfully resumed from {args.resume}, continuing from epoch {start_epoch}")
    elif args.resume is not None:
        if is_main:
            logging.warning(f"Resume checkpoint not found at {args.resume}")

    # ---- optional warn for non-contiguous params ----
    for name, param in model.named_parameters():
        if param.requires_grad and (param.stride() != param.contiguous().stride()):
            if is_main:
                print(f"[Warning] Param '{name}' has non-contiguous strides: {param.shape}, {param.stride()}")

    # ====================== training loop ======================
    for epoch in range(start_epoch, start_epoch + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        
        model.eval()
        total_psnr = 0.0
        total_rmse = 0.0
        total_ssim = 0.0
        tbar = tqdm(valloader, ncols=100, desc=f'[Val] Epoch {epoch}', disable=not is_main)
        image_path = 'fast8x_image'
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
    parser.add_argument('--root_path', type=str, default='../data/ACDC')
    parser.add_argument('--exp', type=str, default='mamba_unrolled')
    parser.add_argument('--dataset', type=str, default='ixi')
    parser.add_argument('--model', type=str, default='mamba_unrolled')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--cfg', type=str, default="../code/configs/vmamba_tiny.yaml")
    parser.add_argument('--opts', default=None, nargs='+')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'])
    parser.add_argument('--resume', default='/scratch/pf2m24/projects/MRIRecon/MambaReconV2/model/mamba_unrolled_140_Patch_1_fastmri_p1_8x2_8x_final/mamba_unrolled/mamba_unrolled_best_psnr_model.pth')
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'])
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--max_iterations', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--deterministic', type=bool, default=True)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--labeled_num', type=int, default=140)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--name', type=str, default='fastmri_8x_test_name')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='by-iter warmup epochs')
    args = parser.parse_args()
    main_worker(args)
