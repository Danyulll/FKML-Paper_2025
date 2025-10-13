#!/usr/bin/env python3
"""
DDQN Inference for KDSS-based FCM on Mixed-Type Data

USAGE:
    python3 test.py --data_csv <path> --init_membership_csv <path> --true_post_csv <path>

DESCRIPTION:
    Loads a pre-trained DDQN agent and uses it to select optimal genetic algorithm parameters
    for kernel density-based fuzzy clustering on mixed-type data.

REQUIREMENTS:
    - Pre-trained agent: ./checkpoints/ddqn_agent.pt
    - Input datasets: CSV files with data, initial membership, and true posterior
"""

import os
import sys
import argparse
import numpy as np
import concurrent.futures as _fut
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as _mp
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

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
# Worker process context
# -------------------------
_CTX = {
    'X': None,
    'C': None,
    'con_idx': None,
    'fac_idx': None,
    'ord_idx': None,
    'U_init': None,
    'U_true': None,
    'epsilon': None,
    'max_iter_obj': None,
    'agent': None,
}


def _worker_init(X, C, con_idx, fac_idx, ord_idx, U_init, U_true, epsilon, max_iter_obj, agent_path):
    # Initialize heavy globals once per process
    _CTX['X'] = X
    _CTX['C'] = C
    _CTX['con_idx'] = con_idx
    _CTX['fac_idx'] = fac_idx
    _CTX['ord_idx'] = ord_idx
    _CTX['U_init'] = U_init
    _CTX['U_true'] = U_true
    _CTX['epsilon'] = epsilon
    _CTX['max_iter_obj'] = max_iter_obj
    # Load agent on CPU to avoid GPU contention across processes
    device = torch.device('cpu')
    agent = DDQNAgent(lr=1e-3, gamma=0.9, epsilon=1.0, device=device)
    agent.load(agent_path)
    _CTX['agent'] = agent


def _run_single_restart(task):
    # task: (m_value, restart_index)
    m_val, _ = task
    X = _CTX['X']; C = _CTX['C']
    con_idx = _CTX['con_idx']; fac_idx = _CTX['fac_idx']; ord_idx = _CTX['ord_idx']
    U_init = _CTX['U_init']; U_true = _CTX['U_true']
    epsilon = _CTX['epsilon']; max_iter_obj = _CTX['max_iter_obj']
    agent = _CTX['agent']

    V, U, reward_history, bw_final, U_history = kdml_fcm(
        X=X, C=C, m=m_val, epsilon=epsilon,
        con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx,
        U_init=U_init, max_iter=100, verbose=False,
        RL_agent=agent, train_B=0, max_iter_obj=max_iter_obj, ckpt_dir="./checkpoints",
        show_fcm_progress=False
    )
    fari_score = fari(U, U_true)
    iter_fari = []
    for t in range(1, len(U_history)):
        prev_U = U_history[t-1]
        cur_U = U_history[t]
        iter_fari.append(fari(prev_U, cur_U))
    # Return minimal data to main proc
    return (float(m_val), float(fari_score), iter_fari)

# -------------------------
# Inference-only DDQN agent
# -------------------------
class DDQNAgent:
    def __init__(self, state_size=25, action_size=25, hidden_dims=[64,64],
                 lr=1e-3, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=32, target_update_freq=100, device=None):
        self.state_size=state_size; self.action_size=action_size
        self.gamma=gamma
        self.epsilon=epsilon; self.epsilon_min=epsilon_min; self.epsilon_decay=epsilon_decay
        self.batch_size=batch_size; self.target_update_freq=target_update_freq
        self.update_counter=0
        self.device = device or pick_device()

        self.online_net = QNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net = QNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict()); self.target_net.eval()

        self.optimizer = None  # No optimizer needed for inference
        self.replay = None     # No replay buffer needed for inference

    def get_state(self, improvement, diversity):
        return (discretize_improvement(improvement), discretize_diversity(diversity))

    def _encode(self, state):
        return onehot_from_index(state_to_index(state), self.state_size)

    def choose_action(self, state):
        # Always use the network (no epsilon-greedy for inference)
        s = torch.from_numpy(self._encode(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(s)
        return int(q.argmax(dim=1))

    def store(self, s, a, r, s2, done):
        pass  # No-op for inference

    def update_network(self):
        pass  # No-op for inference

    def step(self, s, a, r, s2, done=False, metrics_logger=None):
        # No-op for inference; accept metrics_logger for API compatibility
        pass

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.online_net.to(self.device); self.target_net.to(self.device)

# -------------------------
# Main inference function
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="DDQN inference for KDSS-FCM")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to data CSV")
    parser.add_argument("--init_membership_csv", type=str, required=True, help="Path to init membership CSV")
    parser.add_argument("--true_post_csv", type=str, required=True, help="Path to true posterior CSV")
    
    parser.add_argument("--use_cols", type=str, default=None,
        help="Comma-separated 0-based column indices to use from the data CSV (e.g., '0,1,2'). "
             "If omitted, all columns are used.")
    parser.add_argument("--con_idx", type=str, default=None,
        help="Comma-separated indices (relative to the USED columns) that are continuous.")
    parser.add_argument("--fac_idx", type=str, default=None,
        help="Comma-separated indices (relative to the USED columns) that are NOMINAL (unordered categorical).")
    parser.add_argument("--ord_idx", type=str, default=None,
        help="Comma-separated indices (relative to the USED columns) that are ORDINAL (ordered categorical).")

    parser.add_argument("--m", type=float, default=1.2, help="Fuzzifier parameter")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Convergence threshold")
    parser.add_argument("--max_iter_obj", type=int, default=1, help="Inner FCM lookahead for GA objective")
    parser.add_argument("--agent_path", type=str, default="./checkpoints/ddqn_agent.pt", help="Path to trained agent")
    parser.add_argument("--outputs_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--restarts", type=int, default=10, help="Number of KDSS-FCM restarts per m (keep best by FARI)")
    parser.add_argument("--workers", type=int, default=None,
        help="Max worker processes for parallel runs (default: 100)")
    
    args = parser.parse_args()

    ensure_dir(args.outputs_dir)

    # 1) Load CSVs
    print(f"\n=== Loading Dataset ===")
    print(f"Data CSV: {args.data_csv}")
    print(f"Init membership CSV: {args.init_membership_csv}")
    print(f"True posterior CSV: {args.true_post_csv}")
    
    df_full = pd.read_csv(args.data_csv)
    if args.use_cols:
        idxs = parse_index_list(args.use_cols)
        df = df_full.iloc[:, idxs].copy()
        feature_names = [df_full.columns[i] for i in idxs]
        print(f"[Using columns] indices={idxs} -> names={feature_names}")
    else:
        df = df_full.copy()
        feature_names = list(df_full.columns)

    U_init = pd.read_csv(args.init_membership_csv).to_numpy(dtype=float)
    U_true = pd.read_csv(args.true_post_csv).to_numpy(dtype=float)

    print(f"Data shape (raw/used): {df_full.shape} / {df.shape}")
    print(f"Init membership shape: {U_init.shape}")
    print(f"True posterior shape : {U_true.shape}")

    # 2) Indices for feature types (RELATIVE TO USED COLUMNS)
    con_idx = parse_index_list(args.con_idx)
    fac_idx = parse_index_list(args.fac_idx)
    ord_idx = parse_index_list(args.ord_idx)
    if con_idx is None and fac_idx is None and ord_idx is None:
        con_idx, fac_idx, ord_idx = infer_column_types(df)
        print(f"[Auto] con_idx={con_idx}, fac_idx={fac_idx}, ord_idx={ord_idx}")
    else:
        print(f"[Manual] con_idx={con_idx}, fac_idx={fac_idx}, ord_idx={ord_idx}")

    # 3) Encode mixed dataframe -> numeric matrix for kernels
    X = encode_mixed_df(df, fac_idx=fac_idx, ord_idx=ord_idx).to_numpy(dtype=float)

    # 4) Shape checks / cluster count
    N, D = X.shape
    C = U_init.shape[1]
    print(f"Encoded data shape: {X.shape}")
    print(f"Number of clusters: {C}")

    # 5) Load trained DDQN agent
    device = pick_device()
    agent = DDQNAgent(lr=1e-3, gamma=0.9, epsilon=1.0, device=device)
    
    if not os.path.exists(args.agent_path):
        print(f"ERROR: Agent file not found: {args.agent_path}")
        print("Please train the agent first using train.py")
        sys.exit(1)
    
    agent.load(args.agent_path)
    print(f"[DDQN] Loaded trained agent from {args.agent_path}")
    print(f"[DDQN] Using device: {device}")

    # 6) Grid search over m parameter
    m_values = np.linspace(1.1, 2.0, 10)
    fari_scores = []
    
    print(f"\n=== Grid Search over m parameter ===")
    print(f"Testing m values: {m_values}")
    
    # Parallel execution over (m, restart) pairs
    tasks = [(float(m), int(r)) for m in m_values for r in range(args.restarts)]
    # Default to using 100 workers unless explicitly overridden.
    requested_workers = int(max(1, args.workers)) if args.workers is not None else 100
    max_workers = min(requested_workers, len(tasks))

    print(f"[Parallel] Initializing {max_workers} workers for {len(tasks)} tasks (m-values x restarts)")
    results_by_m = {float(m): [] for m in m_values}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(
            X,
            C,
            con_idx,
            fac_idx,
            ord_idx,
            U_init,
            U_true,
            args.epsilon,
            args.max_iter_obj,
            args.agent_path,
        ),
    ) as ex:
        futures = {ex.submit(_run_single_restart, t): t for t in tasks}
        for _ in tqdm(_fut.as_completed(futures), total=len(futures), desc="Parallel runs"):
            pass
    # Collect results (iterate futures again to extract results in a deterministic pass)
    for fut, (m_task, _) in futures.items():
        m_val, fari_score_val, iter_series = fut.result()
        results_by_m[m_val].append((fari_score_val, iter_series))

    # Aggregate per-m results, plot, and store FARI scores
    for m in m_values:
        m_key = float(m)
        per_restart = results_by_m[m_key]
        if not per_restart:
            fari_scores.append(float('-inf'))
            continue
        best_fari_m = max(fr for fr, _ in per_restart)
        per_restart_fari_series = [series for _, series in per_restart]
        fari_scores.append(best_fari_m)
        if args.verbose:
            print(f"m={m:.2f}: best FARI over {args.restarts} restarts = {best_fari_m:.4f}")

        fig, ax = plt.subplots(figsize=(10,6))
        for ridx, series in enumerate(per_restart_fari_series):
            ax.plot(range(1, len(series)+1), series, label=f"restart {ridx+1}", alpha=0.7)
        ax.set_title(f"Per-iteration FARI(prev,cur) for m={m:.3f}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("FARI(prev, cur)")
        ax.grid(True, alpha=0.3)
        if len(per_restart_fari_series) <= 12:
            ax.legend()
        plt.tight_layout()
        m_dir = os.path.join(args.outputs_dir, f"m_{m:.3f}")
        ensure_dir(m_dir)
        plt.savefig(os.path.join(m_dir, "fari_between_iterations.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 7) Find best m and results
    best_idx = np.argmax(fari_scores)
    best_m = m_values[best_idx]
    best_fari = fari_scores[best_idx]
    
    print(f"\n=== Results ===")
    print(f"Best m: {best_m:.3f}")
    print(f"Best FARI: {best_fari:.4f}")
    print(f"All FARI scores: {[f'{f:.4f}' for f in fari_scores]}")

    # 8) Plot FARI vs m
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, fari_scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=best_m, color='red', linestyle='--', alpha=0.7, label=f'Best m = {best_m:.3f}')
    plt.axhline(y=best_fari, color='red', linestyle='--', alpha=0.7, label=f'Best FARI = {best_fari:.4f}')
    plt.xlabel('Fuzzifier Parameter (m)')
    plt.ylabel('FARI Score')
    plt.title('FARI vs Fuzzifier Parameter (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(args.outputs_dir, "fari_vs_m.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {plot_path}")

    # 9) Save results to text file
    results_path = os.path.join(args.outputs_dir, "best_configuration.txt")
    with open(results_path, 'w') as f:
        f.write("DDQN-KDSS-FCM Results\n")
        f.write("=====================\n\n")
        f.write(f"Dataset: {args.data_csv}\n")
        f.write(f"Data shape: {X.shape}\n")
        f.write(f"Number of clusters: {C}\n")
        f.write(f"Feature types: con_idx={con_idx}, fac_idx={fac_idx}, ord_idx={ord_idx}\n\n")
        f.write(f"Best m: {best_m:.6f}\n")
        f.write(f"Best FARI: {best_fari:.6f}\n\n")
        f.write("All results:\n")
        for m, fari_val in zip(m_values, fari_scores):
            f.write(f"  m={m:.3f}: FARI={fari_val:.6f}\n")
    
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
