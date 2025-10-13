#!/bin/bash
#SBATCH --job-name=big-dataset
#SBATCH --account=def-thompsjr   # <-- change this
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1                 # or --gpus-per-node=v100:1 / a100:1 if you need a type
#SBATCH --cpus-per-task=100
#SBATCH --mem=32G
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
  --data_csv "./data/big dataset/bigdataset_data.csv" \
  --init_membership_csv "./data/big dataset/bigdataset_membership.csv" \
  --true_post_csv "./data/big dataset/bigdata_truepost.csv" \
  --restarts 10 \
  --workers 100 \
  --outputs_dir "./outputs/dataset_big" \
  --verbose \
  --use_cols "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14" \
  --con_idx "0,1,2,3,4,5,6,10,11,12 \
  --ord_idx "7,8,9,13,14"

