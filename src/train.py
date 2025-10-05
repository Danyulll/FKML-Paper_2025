#!/usr/bin/env python3
"""
DDQN Training for KDSS-based FCM on Mixed-Type Data

USAGE:
    python3 train.py

DESCRIPTION:
    Trains a Double Deep Q-Network (DDQN) agent to learn optimal genetic algorithm parameters
    for kernel density-based fuzzy clustering on mixed-type data (continuous + ordinal features).

ALGORITHM PARAMETERS:
    - State Space: 25 discrete states (5×5 grid: improvement × diversity levels)
    - Action Space: 25 actions (5×5 grid: p_m × p_c GA parameters)
    - Reward System: Based on GA improvement and population diversity
    - Training Episodes: 150 (configurable)
    - Network: 2-layer MLP (64→64→25) with experience replay
    - Epsilon Decay: 1.0 → 0.05 over training

TRAINING DATASET:
    - 3D mixed-type data: 2 continuous (Gaussian+Gamma) + 1 ordinal (Poisson)
    - 3 clusters, 800 samples
    - Files: CUSTOM_G_mean_-3_0_3_sd_1_1_1_Ga_a_2_4_6_b_1.5_1.2_0.9_Pois_2_6_10_(train).csv

OUTPUTS:
    - ./checkpoints/ddqn_agent.pt (trained agent)
    - ./outputs/ddqn_training_rewards.png (training progress)
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import all model components from model.py
from model import (
    pick_device, ensure_dir, parse_index_list, infer_column_types, encode_mixed_df,
    fari, dkss_distance, kdml_fcm_no_bw_update, objective, compute_diversity,
    clamp_continuous, tournament_selection, one_point_crossover, mutate, genetic_algorithm,
    REWARD_TABLE, discretize_improvement, discretize_diversity, state_to_index,
    onehot_from_index, index_to_action, QNetwork, DDQNAgent, estimate_bandwidth_vector_RL,
    train_RL_agent, centers_from_membership, kdml_fcm
)

# -------------------------
# Main training function
# -------------------------
def main():
    # Default training configuration - no command line arguments needed
    print("=== DDQN Training for KDSS-FCM ===")
    print("Using default training dataset and parameters")
    
    # Training dataset paths (hardcoded)
    data_csv = "./data/3D 2 Cont 1 Ord C=2/CUSTOM_G_mean_-3_0_3_sd_1_1_1_Ga_a_2_4_6_b_1.5_1.2_0.9_Pois_2_6_10_(train).csv"
    init_membership_csv = "./data/3D 2 Cont 1 Ord C=2/CUSTOM_G_mean_-3_0_3_sd_1_1_1_Ga_a_2_4_6_b_1.5_1.2_0.9_Pois_2_6_10_membership_(train).csv"
    true_post_csv = "./data/3D 2 Cont 1 Ord C=2/CUSTOM_G_mean_-3_0_3_sd_1_1_1_Ga_a_2_4_6_b_1.5_1.2_0.9_Pois_2_6_10_true_posterior_(train).csv"
    
    # Training parameters (hardcoded)
    use_cols = None  # Use all columns
    con_idx = [0, 1]  # First two columns are continuous
    fac_idx = None    # No nominal features
    ord_idx = [2]     # Third column is ordinal
    m = 1.2
    epsilon = 1e-3
    max_iter_obj = 1
    train_B = 150
    ckpt_dir = "./checkpoints"
    outputs_dir = "./outputs"
    verbose = True

    ensure_dir(ckpt_dir)
    ensure_dir(outputs_dir)

    # 1) Load CSVs
    print(f"\n=== Training Dataset ===")
    print(f"Training data CSV: {data_csv}")
    print(f"Init membership CSV: {init_membership_csv}")
    print(f"True posterior CSV: {true_post_csv}")
    
    df_full = pd.read_csv(data_csv)
    if use_cols:
        idxs = parse_index_list(use_cols)
        df = df_full.iloc[:, idxs].copy()
        feature_names = [df_full.columns[i] for i in idxs]
        print(f"[Using columns] indices={idxs} -> names={feature_names}")
    else:
        df = df_full.copy()
        feature_names = list(df_full.columns)

    U_init = pd.read_csv(init_membership_csv).to_numpy(dtype=float)
    U_true = pd.read_csv(true_post_csv).to_numpy(dtype=float)

    print(f"Data shape (raw/used): {df_full.shape} / {df.shape}")
    print(f"Init membership shape: {U_init.shape}")
    print(f"True posterior shape : {U_true.shape}")

    # 2) Indices for feature types (RELATIVE TO USED COLUMNS)
    # Already set above: con_idx = [0, 1], fac_idx = None, ord_idx = [2]
    print(f"[Training] con_idx={con_idx}, fac_idx={fac_idx}, ord_idx={ord_idx}")

    # 3) Encode mixed dataframe -> numeric matrix for kernels
    X = encode_mixed_df(df, fac_idx=fac_idx, ord_idx=ord_idx).to_numpy(dtype=float)

    # 4) Shape checks / cluster count
    N, D = X.shape
    C = U_init.shape[1]
    print(f"Encoded data shape: {X.shape}")
    print(f"Number of clusters: {C}")

    # 5) Initialize DDQN agent
    device = pick_device()
    agent = DDQNAgent(lr=1e-3, gamma=0.9, epsilon=1.0, device=device)
    print(f"[DDQN] Initialized on device: {device}")

    # 6) Random initial bandwidths
    D = X.shape[1]
    initial_bw = []
    for j in range(D):
        if con_idx and j in con_idx:
            initial_bw.append(np.random.uniform(0.1, 100.0))
        elif (fac_idx and j in fac_idx) or (ord_idx and j in ord_idx):
            initial_bw.append(np.random.uniform(0.0, 1.0))
        else:
            initial_bw.append(np.random.uniform(0.1, 10.0))

    # 7) Train the DDQN agent
    print(f"\n=== Training DDQN Agent for {train_B} episodes ===")
    agent, rewards = train_RL_agent(
        X=X, initial_bw=initial_bw, RL_agent=agent, prev_ga_obj=None,
        U_ref=U_init, C=C, m=m, con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx,
        B=train_B, max_iter_obj=max_iter_obj, ckpt_dir=ckpt_dir, verbose=verbose
    )

    # 8) Save final agent
    agent_path = os.path.join(ckpt_dir, "ddqn_agent.pt")
    agent.save(agent_path)
    print(f"[Agent] Saved to {agent_path}")

    # 9) Final evaluation
    print("\n=== Training Complete ===")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total rewards: {len(rewards)}")
    print(f"Mean reward: {np.mean(rewards):.4f}")
    print(f"Final cumulative reward: {np.sum(rewards):.4f}")

if __name__ == "__main__":
    main()
