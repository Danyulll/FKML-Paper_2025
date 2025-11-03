#!/bin/bash
#SBATCH --job-name=2D-Gaussian-C-2-(2)
#SBATCH --account=def-thompsjr   # <-- change this
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out

set -euo pipefail

# Load Python (adjust to what's available on your site)
module load python/3.10

# Activate your venv (replace with your path)
source "$HOME/tf-venv/bin/activate"

echo "=== Node & GPU info ==="
hostname
nvidia-smi || true
echo "======================= "

python3 ./src/test.py \
  --data_csv "./data/2D Gaussian C=2 (2)/2D_G_mean_-2.5_2.5_sd_1_1_G_mean_3_3_sd_1.1_1.1.csv" \
  --init_membership_csv "./data/2D Gaussian C=2 (2)/2D_G_mean_-2.5_2.5_sd_1_1_G_mean_3_3_sd_1.1_1.1_membership.csv" \
  --true_post_csv "./data/2D Gaussian C=2 (2)/2D_G_mean_-2.5_2.5_sd_1_1_G_mean_3_3_sd_1.1_1.1_true_posterior.csv" \
  --restarts 10 \
  --outputs_dir "./outputs/dataset_2D_Gaussian_2" \
  --verbose \
  --con_idx "0,1" \
  --use_cols "0,1"