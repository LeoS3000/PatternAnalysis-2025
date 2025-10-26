#!/bin/bash

# --- Slurm Job Directives ---
#SBATCH --job-name=unet3D
#SBATCH --partition=comp3710  
#SBATCH --time=04:20:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=l.strietzel@student.uq.edu.au

echo "=========================================================="
echo "Starting UNet3D Sanity Check"
echo "=========================================================="

# 1. Initialize Conda
source /home/Student/s4979785/miniconda/etc/profile.d/conda.sh
conda activate brains-gan-new

# 2. Define the project directory
PROJECT_DIR=$SLURM_SUBMIT_DIR

python ${PROJECT_DIR}/train.py \
    --epochs 100 \
    --batch_size 4 \
    --checkpoint_dir ${PROJECT_DIR}/checkpoints_test

echo "=========================================================="
echo "Job finished"
echo "=========================================================="