#!/bin/bash
#SBATCH -p swarm_h100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=train_V1_4_P1
#SBATCH --output=logs/V1_4_pro_P1.log
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=80G

source ~/.bashrc
conda activate Mamba_IR

cd /scratch/pf2m24/projects/MRIRecon/MambaRecon/code

echo "CPUS ON NODE: $SLURM_CPUS_ON_NODE"
echo "JOB CPUS PER NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "CPUS PER TASK: $SLURM_CPUS_PER_TASK"

export PYTHONPATH=/scratch/pf2m24/projects/MRIRecon/MambaReconV3/code:$PYTHONPATH

torchrun --nproc_per_node=2 --master_port=29513 train_scan_out_4_pro.py --batch_size 2 --patch_size 1 --name v1_prostate_4_P1