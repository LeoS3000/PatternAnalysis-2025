#!/bin/bash

# --- Slurm Job Directives ---
#SBATCH --job-name=cifar10_test       # Name for your job
#SBATCH --partition=test              # Request the 'test' partition
#SBATCH --time=00:20:00               # Max wall-clock time (must be <= 20 mins for 'test')
#SBATCH --nodes=1                     # Request a single node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # Request 4 CPU cores for data loaders
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --mail-type=END,FAIL          # Send email on job END or FAIL
#SBATCH --mail-user=l.strietzel@student.uq.edu.au

# --- Commands to be Executed ---

echo "=========================================================="
echo "Starting on $(hostname)"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================================="

# 1. Initialize Conda
# This is the corrected line:
source /home/Student/s4979785/miniconda/etc/profile.d/conda.sh

# 2. Activate your Conda environment
conda activate brains-gan

# 3. Run your Python training script
# The 'train_cifar10.py' file should be in the same directory you submit from
python train.py \
    --epochs 100 \
    --warmup_epochs 5 \
    --batch_size 512 \
    --base_lr 0.1 \
    --num_workers $SLURM_CPUS_PER_TASK

echo "=========================================================="
echo "Job finished"
echo "=========================================================="