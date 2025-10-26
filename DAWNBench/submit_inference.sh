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

PROJECT_DIR=$SLURM_SUBMIT_DIR

# 1. Initialize Conda
# This is the corrected line:
source /home/Student/s4979785/miniconda/etc/profile.d/conda.sh

# 2. Activate your Conda environment
conda activate brains-gan

# --- Run the Inference Script ---
# Ensure the checkpoint file from your training job exists
CHECKPOINT_FILE="resnet18_cifar10.pth"

if [ ! -f ${PROJECT_DIR}"/$CHECKPOINT_FILE" ]; then
    echo "Error: Checkpoint file '$CHECKPOINT_FILE' not found!"
    echo "Please run the training job first to create the model file."
    exit 1
fi

echo "Running Python inference script on '${PROJECT_DIR}/$CHECKPOINT_FILE'..."

python inference.py --checkpoint "${PROJECT_DIR}/$CHECKPOINT_FILE"

echo "=========================================================="
echo "Inference job finished at $(date)"
echo "=========================================================="