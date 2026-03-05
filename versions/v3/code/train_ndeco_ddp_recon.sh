#!/bin/bash
#SBATCH -p swarm_h100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=train_vmamba
#SBATCH --output=logs/cc359_p2_8x2_moe_8x_1103_loss_0.05_mean_top1_fpc.log
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=80G

source ~/.bashrc
conda activate Mamba_IR

cd /scratch/pf2m24/projects/MRIRecon/MambaReconV3/code

echo "CPUS ON NODE: $SLURM_CPUS_ON_NODE"
echo "JOB CPUS PER NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "CPUS PER TASK: $SLURM_CPUS_PER_TASK"

export PYTHONPATH=/scratch/pf2m24/projects/MRIRecon/MambaReconV3/code:$PYTHONPATH

torchrun --nproc_per_node=2 --master_port=29518 train_scan_out_8_cc359_best_ddp_loss.py --batch_size 4 --name cc359_p2_8x2_moe_fpc_8x