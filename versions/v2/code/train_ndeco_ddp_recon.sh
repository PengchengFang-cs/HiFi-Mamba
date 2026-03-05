#!/bin/bash
#SBATCH -p swarm_h100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=train_vmamba
#SBATCH --output=logs/cc359_p1_8x2_dw5_final_8x_0828.log
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G

source ~/.bashrc
conda activate Mamba_IR

cd /scratch/pf2m24/projects/MRIRecon/MambaReconV2/code

echo "CPUS ON NODE: $SLURM_CPUS_ON_NODE"
echo "JOB CPUS PER NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "CPUS PER TASK: $SLURM_CPUS_PER_TASK"

export PYTHONPATH=/scratch/pf2m24/projects/MRIRecon/MambaReconV2/code:$PYTHONPATH

torchrun --nproc_per_node=2 --master_port=29513 train_scan_out_8_cc359_best_ddp.py --batch_size 2 --patch_size 1 --name cc359_p1_8x2_dw5_final_8x_0902