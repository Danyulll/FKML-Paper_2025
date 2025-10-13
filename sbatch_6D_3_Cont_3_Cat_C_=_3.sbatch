#!/bin/bash
#SBATCH --job-name=6D-3-Cont-3-Cat-C-3
#SBATCH --account=def-thompsjr   # <-- change this
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1                 # or --gpus-per-node=v100:1 / a100:1 if you need a type
#SBATCH --cpus-per-task=100
#SBATCH --mem=16G
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
  --data_csv "./data/6D 3 Cont 3 Cat C = 3/CUSTOM_G_mean_-4_0_4_sd_1_1_1_Ga_a_2_4_8_b_1.5_1.2_0.9_Pois_1_9_15_NB_r_8_8_8_p_0.65_0.5_0.35_NoiseC1_NoiseD1.csv" \
  --init_membership_csv "./data/6D 3 Cont 3 Cat C = 3/CUSTOM_G_mean_-4_0_4_sd_1_1_1_Ga_a_2_4_8_b_1.5_1.2_0.9_Pois_1_9_15_NB_r_8_8_8_p_0.65_0.5_0.35_NoiseC1_NoiseD1_membership.csv" \
  --true_post_csv "./data/6D 3 Cont 3 Cat C = 3/CUSTOM_G_mean_-4_0_4_sd_1_1_1_Ga_a_2_4_8_b_1.5_1.2_0.9_Pois_1_9_15_NB_r_8_8_8_p_0.65_0.5_0.35_NoiseC1_NoiseD1_true_posterior.csv" \
  --restarts 10 \
  --workers 100 \
  --outputs_dir "./outputs/dataset_6D_3Cont_3Cat_C3" \
  --verbose \
  --use_cols "0,1,2,3,4,5" \
  --con_idx "0,1,4" \
  --ord_idx "2,3,5"

