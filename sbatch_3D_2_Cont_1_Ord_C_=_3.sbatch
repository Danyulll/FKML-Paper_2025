#!/bin/bash
#SBATCH --job-name=3D 2 Cont 1 Ord C = 3
#SBATCH --account=def-thompsjr   # <-- change this
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1                 # or --gpus-per-node=v100:1 / a100:1 if you need a type
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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
  --data_csv "./data/3D 2 Cont 1 Ord C = 3/3D_Be_a_1.5_3_4.5_b_4.5_3_1.5_Ln_m_-0.2_0.2_0.6_s_0.4_0.4_0.4_NB_r_3_5.5_8_p_0.65_0.5_0.35.csv" \
  --init_membership_csv "./data/3D 2 Cont 1 Ord C = 3/3D_Be_a_1.5_3_4.5_b_4.5_3_1.5_Ln_m_-0.2_0.2_0.6_s_0.4_0.4_0.4_NB_r_3_5.5_8_p_0.65_0.5_0.35_membership.csv" \
  --true_post_csv "./data/3D 2 Cont 1 Ord C = 3/3D_Be_a_1.5_3_4.5_b_4.5_3_1.5_Ln_m_-0.2_0.2_0.6_s_0.4_0.4_0.4_NB_r_3_5.5_8_p_0.65_0.5_0.35_true_posterior.csv" \
  --restarts 10 \
  --outputs_dir "./outputs/dataset_3D_2Cont_1Ord_C3" \
  --verbose \
  --use_cols "0,1,2" \
  --con_idx "0,1" \
  --ord_idx "2"