#!/bin/bash
#SBATCH --job-name=vae_train                # descriptive name
#SBATCH --output=%x-%j.out                 # stdout
#SBATCH --error=%x-%j.err                  # stderr
#SBATCH --partition=a100-test               # quick‑dev queue (20 min limit)
#   For longer runs use:   #SBATCH --partition=a100
#SBATCH --gres=gpu:1                       # request 1 GPU (or 1 shard)
#SBATCH --cpus-per-task=8                  # number of CPU cores
#SBATCH --mem=32G                          # RAM per node (adjust if needed)
#SBATCH --time=00:20:00                    # 20 min for test; increase for real run
#SBATCH --requeue                          # allow Slurm to requeue after pre‑empt
#SBATCH --signal=B:SIGTERM@120             # send SIGTERM 120 s before kill

# --------------------------------------------------------------
# Load the environment that the cluster provides
# --------------------------------------------------------------
module purge
module load python/3.11               # version that ships with the cluster
module load cuda/12.2                  # for A100 nodes (adjust if needed)
module load pytorch/2.2.0               # includes torch, torchvision, etc.

# --------------------------------------------------------------
# Activate your conda / virtualenv (if you use one)
# --------------------------------------------------------------
# Example with conda (replace `brains-gan` with your env name)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate brains-gan

# --------------------------------------------------------------
# Run the training script
# --------------------------------------------------------------
srun python train_vae_slurm.py