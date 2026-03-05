from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import math
import argparse
import logging
import os
import random
import shutil
import sys


from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config

from dataloaders.dataset import ixi_dataset, fastmri_dataset
from val_2D import test_single_slice

from dataloaders.fastMRI_dataset_PGIUN import fastMRI_dataset
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='mamba_unrolled', help='experiment_name')

parser.add_argument('--dataset', type=str,
                    default='fastmri', help='dataset name')

parser.add_argument('--model', type=str,
                    default='mamba_unrolled', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')


parser.add_argument(
    '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')


parser.add_argument('--max_iterations', type=int,
                    default=100000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=int,  default=2,
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=140,
                    help='labeled data')
parser.add_argument('--gpu_id', type=int,  default=0)

args = parser.parse_args()

print("dataset: ", args.dataset)

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

def adjust_learning_rate(optimizer, epoch, total_epochs=100, lr=1e-3, warmup_epochs=5, min_lr=1e-6):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) /
             (total_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def train(args, snapshot_path):
    torch.autograd.set_detect_anomaly(True)
    base_lr = args.base_lr
    print("learning rate: ", base_lr)
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = VIM_seg(config, patch_size=args.patch_size,
                    num_classes=2).cuda()
    # model.load_from(config)

    # if args.dataset == "ixi":

    #     db_train = ixi_dataset(split="train")

    #     db_val = ixi_dataset(split="val")

    # elif args.dataset == "fastmri":
    #     db_train = fastmri_dataset(split="train")

    #     db_val = fastmri_dataset(split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
    #                          num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    # valloader = DataLoader(db_val, batch_size=1, shuffle=False,
    #                        num_workers=1)

    train_dataset = fastMRI_dataset(
        data_dir="/scratch/pf2m24/data/FastMRI/singlecoil_train",
        acceleration=8,
        mask_type ='equispaced',
        resolution = 320,
        type="train_",
    )
    val_dataset = fastMRI_dataset(
        data_dir="/scratch/pf2m24/data/FastMRI/singlecoil_val",
        acceleration=8,
        mask_type ='equispaced',
        resolution = 320,
        type="val",
    )
    trainloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=32,
                            pin_memory=True,
                            prefetch_factor=4)
    valloader = DataLoader(val_dataset,
                        batch_size= args.batch_size,
                        shuffle=False,
                        num_workers=32,
                        pin_memory=True,
                        prefetch_factor=4)

    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=base_lr)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    # max_epoch = max_iterations // len(trainloader) + 1
    # best_performance = 0.0
    max_epoch = 100
    best_psnr_epoch = 0
    best_ssim_epoch = 0
    best_psnr = -1
    best_ssim = -1
    # iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in range(max_epoch):
        pbar = tqdm(
            trainloader,
            ncols=140,
            desc=f'Epoch {epoch}/{max_epoch}',
            unit='batch',
            leave=True
        )
        model.train()
        for i_batch, sampled_batch in enumerate(pbar):
            us_image, fs_target, us_mask, coil_maps = sampled_batch['us_image'], sampled_batch[
                'fs_image'], sampled_batch['us_mask'], sampled_batch['coil_map']

            us_image, fs_target, us_mask, coil_maps = us_image.cuda(
            ), fs_target.cuda(), us_mask.cuda(), coil_maps.cuda()

            outputs = model(us_image, us_mask, coil_maps)
            outputs = torch.abs(outputs[:, 0, :, :] + 1j * outputs[:, 1, :, :])
            fs_target = torch.abs(fs_target[:, 0, :, :] + 1j * fs_target[:, 1, :, :])
            loss = torch.mean(torch.abs(outputs - fs_target))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = adjust_learning_rate(optimizer, epoch+1)

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            pbar.set_postfix({'loss': loss.item()})

        model.eval()
        total_psnr = total_rmse = total_ssim = total_l1 = 0.0
        tbar = tqdm(valloader, ncols=130, desc=f'Validate {epoch+1}')
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(tbar):
                us_image, fs_target, us_mask, coil_maps = sampled_batch['us_image'], sampled_batch[
                    'fs_image'], sampled_batch['us_mask'], sampled_batch['coil_map']

                us_image, fs_target, us_mask, coil_maps = us_image.cuda(
                ), fs_target.cuda(), us_mask.cuda(), coil_maps.cuda()
                
                outputs = model(us_image, us_mask, coil_maps)
                outputs = torch.abs(outputs[:, 0, :, :] + 1j * outputs[:, 1, :, :]).unsqueeze(1)

                fs_target = torch.abs(
                    fs_target[:, 0, :, :] + 1j * fs_target[:, 1, :, :]).unsqueeze(1)
                

                total_psnr += peak_signal_noise_ratio(fs_target.detach().cpu().numpy(), outputs.clamp(0, 1).detach().cpu().numpy(), data_range=fs_target.max())
                total_rmse += nmse(fs_target.detach().cpu().numpy(), outputs.clamp(0, 1).detach().cpu().numpy())
                total_ssim += calculate_ssim(fs_target.detach().cpu().numpy(), outputs.clamp(0, 1).detach().cpu().numpy())
        total_psnr /= len(valloader)
        total_rmse /= len(valloader)
        total_ssim /= len(valloader)

        writer.add_scalar('info/val_{}_psnr'.format(i_batch),
                            total_psnr, iter_num)
        writer.add_scalar('info/val_{}_ssim'.format(i_batch),
                            total_ssim, iter_num)

        writer.add_scalar('info/val_mean_psnr', total_psnr, iter_num)
        writer.add_scalar('info/val_mean_ssim', total_ssim, iter_num)

        if total_psnr > best_psnr:
            best_psnr = total_psnr
            best_psnr_epoch = epoch + 1
            save_mode_path = os.path.join(snapshot_path,
                                            'iter_{}_psnr_{}.pth'.format(
                                                iter_num, round(best_psnr, 4)))
            save_best = os.path.join(snapshot_path,
                                        '{}_best_psnr_model.pth'.format(args.model))
            torch.save(model.state_dict(), save_mode_path)
            torch.save(model.state_dict(), save_best)
        if total_ssim > best_ssim:
            best_ssim = total_ssim
            best_ssim_epoch = epoch + 1
            save_mode_path = os.path.join(snapshot_path,
                                            'iter_{}_ssim_{}.pth'.format(
                                                iter_num, round(best_ssim, 4)))
            save_best = os.path.join(snapshot_path,
                                        '{}_best_ssim_model.pth'.format(args.model))
            torch.save(model.state_dict(), save_mode_path)
            torch.save(model.state_dict(), save_best)
        
        logging.info(f'Slice Epoch {epoch+1}: Validation PSNR: {total_psnr:.4f}, RMSE: {total_rmse:.4f}, SSIM: {total_ssim:.4f}')
        logging.info(f'Best PSNR at epoch {best_psnr_epoch}: {best_psnr:.4f}')
        logging.info(f'Best SSIM at epoch {best_ssim_epoch}: {best_ssim:.4f}')
        model.train()

    writer.close()
    logging.info(f'Best PSNR at epoch {best_psnr_epoch}: {best_psnr:.4f}')
    logging.info(f'Best SSIM at epoch {best_ssim_epoch}: {best_ssim:.4f}')
    return "Training Finished!"

def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2
def calculate_ssim(img1, img2):
    """计算四维张量 (batch_size, channels, width, height) 的 SSIM"""
    ssim_values = []
    batch_size, channels, width, height = img1.shape
    for i in range(batch_size):
        for j in range(channels):
            ssim_value = structural_similarity(img1[i, j], img2[i, j], channel_axis=None, data_range=img1[i, j].max())
            ssim_values.append(ssim_value)
    return np.mean(ssim_values)
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(int(args.gpu_id))
    snapshot_path = "../model/{}_{}_labeled_scan_out/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
