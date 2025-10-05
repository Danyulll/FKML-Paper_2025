#!/usr/bin/env python3
"""
Shared Model Components for DDQN-based KDSS-FCM

This module contains all the shared model code used by both train.py and test.py:
- PyTorch device management
- Data preprocessing utilities
- Kernel density functions
- FCM clustering algorithms
- Genetic algorithm components
- DDQN agent implementation
- Reward system and state management
"""

import os
import sys
import math
import random
from collections import deque
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the RL metrics logger
from rl_metrics import MetricsLogger

# -------------------------
# PyTorch (for DDQN)
# -------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    print("ERROR: This script needs PyTorch (torch) for the DDQN agent.\n"
          "Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu", file=sys.stderr)
    raise

# -------------------------
# Device picker (CUDA / MPS / CPU)
# -------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def parse_index_list(s):
    """Parse comma-separated indices like '0,2,5'. Return list[int] or None if empty/None."""
    if s is None or str(s).strip()=="":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()!=""]

def infer_column_types(df: pd.DataFrame):
    """
    Heuristic:
      - float columns -> continuous
      - int columns with <= 12 unique values -> ordinal
      - object/category -> nominal
    """
    con_idx, fac_idx, ord_idx = [], [], []
    for j, col in enumerate(df.columns):
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            con_idx.append(j)
        elif pd.api.types.is_integer_dtype(s):
            nunq = s.nunique(dropna=True)
            if nunq <= 12:
                ord_idx.append(j)
            else:
                con_idx.append(j)  # high-cardinality ints treated as continuous
        else:
            fac_idx.append(j)
    return con_idx, fac_idx, ord_idx

def encode_mixed_df(df: pd.DataFrame, fac_idx=None, ord_idx=None) -> pd.DataFrame:
    """
    Encode a mixed-type DataFrame into purely numeric columns:
      - Continuous columns: left as-is (but cast to numeric).
      - fac_idx (nominal): converted to unordered categorical codes.
      - ord_idx (ordinal): converted to ordered categorical codes (numeric sort if possible, else lexical), then to codes.
      - Any remaining non-numeric columns: treated as nominal and coded.
    Returns a numeric DataFrame (ints/floats) suitable for distance computations.
    """
    df2 = df.copy()
    n_cols = df2.shape[1]

    fac_idx = [] if fac_idx is None else list(fac_idx)
    ord_idx = [] if ord_idx is None else list(ord_idx)

    # 1) Handle nominal (unordered categorical)
    for j in fac_idx:
        s = df2.iloc[:, j]
        if not isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype("category")
        s = s.cat.as_unordered()
        df2.iloc[:, j] = s.cat.codes.astype("int64")

    # 2) Handle ordinal (ordered categorical)
    for j in ord_idx:
        s = df2.iloc[:, j]
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.cat.as_ordered()
            df2.iloc[:, j] = pd.Series(s).cat.codes.astype("int64")
        else:
            # Try numeric ordering first
            try:
                vals = pd.to_numeric(s, errors="raise")
                cats = np.unique(vals.dropna())
                # Use string categories to be robust to mixed types while preserving order
                cat_strings = [str(c) for c in np.sort(cats)]
                s_str = s.astype(str)
                s_ord = pd.Categorical(s_str, categories=cat_strings, ordered=True)
                df2.iloc[:, j] = pd.Series(s_ord).cat.codes.astype("int64")
            except Exception:
                # Fallback to lexical ordering
                cats = sorted(s.astype(str).dropna().unique())
                s_ord = pd.Categorical(s.astype(str), categories=cats, ordered=True)
                df2.iloc[:, j] = pd.Series(s_ord).cat.codes.astype("int64")

    # 3) Remaining columns: if still non-numeric, treat as nominal codes
    for j in range(n_cols):
        s = df2.iloc[:, j]
        if not pd.api.types.is_numeric_dtype(s):
            s = s.astype("category")
            df2.iloc[:, j] = s.cat.codes.astype("int64")

    # 4) Ensure numeric dtype across the board
    for col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    return df2

# -------------------------
# Categorical cardinalities (for nominal/ordinal kernels)
# -------------------------
CAT_CARD = None  # np.ndarray of length D with per-column unique counts

def set_categorical_cardinalities(X: np.ndarray, fac_idx=None, ord_idx=None):
    """
    Precompute per-column cardinalities from the FULL encoded dataset X.
    Only nominal/ordinal columns matter; others are left as 1.
    """
    global CAT_CARD
    D = X.shape[1]
    card = np.ones(D, dtype=int)
    for j in (fac_idx or []):
        card[j] = int(len(np.unique(X[:, j])))
    for j in (ord_idx or []):
        card[j] = int(len(np.unique(X[:, j])))
    CAT_CARD = card

def make_aligned_bandwidths(D, con_idx=None, fac_idx=None, ord_idx=None):
    """
    Create a length-D bandwidth vector aligned to original columns:
      - continuous j in con_idx:    U(0.1, 100.0)
      - nominal j in fac_idx:       U(0.0, 1.0)
      - ordinal j in ord_idx:       U(0.0, 1.0)
      - anything else:              U(0.1, 10.0)
    """
    con_idx = [] if con_idx is None else list(con_idx)
    fac_idx = [] if fac_idx is None else list(fac_idx)
    ord_idx = [] if ord_idx is None else list(ord_idx)

    bw = np.empty(D, dtype=float)
    for j in range(D):
        if j in con_idx:
            bw[j] = np.random.uniform(0.1, 100.0)
        elif j in fac_idx or j in ord_idx:
            bw[j] = np.random.uniform(0.0, 1.0)
        else:
            bw[j] = np.random.uniform(0.1, 10.0)
    return bw


# -------------------------
# FARI (Adjusted Fuzzy Rand Index)
# -------------------------
def fari(a: np.ndarray, b: np.ndarray):
    """
    Adjusted Fuzzy Rand Index between membership matrices a and b (n x k).
    Returns a single float.
    """
    n = a.shape[0]
    A = a @ a.T
    B = b @ b.T
    j = np.ones((n, n))

    Na = (np.sum(A) / np.sum(A * A)) * A
    Nb = (np.sum(B) / np.sum(B * B)) * B

    ri = (np.sum(Na * Nb) + np.sum((j - Na) * (j - Nb)) - n) / (n * (n - 1))

    M = j / n
    R = np.eye(n) - M

    term1 = (2 * np.sum(A) * np.sum(B)) / (np.sum(A * A) * np.sum(B * B))
    term2 = np.sum(M * A) * np.sum(M * B) + (1 / (n - 1)) * np.sum(R * A) * np.sum(R * B)
    term3 = (np.sum(A) ** 2) / np.sum(A * A)
    term4 = (np.sum(B) ** 2) / np.sum(B * B)
    Eri = (term1 * term2 - term3 - term4 + n**2 - n) / (n * (n - 1))

    ari = (ri - Eri) / (1 - Eri)
    return float(ari)

# -------------------------
# Kernels (continuous / nominal / ordinal)
# -------------------------
def c_gaussian(A, B, bws, Cmatrix_unused):
    return np.sum((2 * np.pi) ** (-0.5) * np.exp(-(((A - B) / bws) ** 2) / 2))

def c_uniform(A, B, bws, Cmatrix_unused):
    return np.sum(np.where(np.abs((A - B) / bws) <= 1, 0.5, 0.0))

def c_epanechnikov(A, B, bws, Cmatrix_unused):
    return np.sum(np.where(np.abs((A - B) / bws) <= 1, 0.75 * (1 - ((A - B) / bws) ** 2), 0.0))

def c_triangle(A, B, bws, Cmatrix_unused):
    return np.sum(np.where(np.abs((A - B) / bws) <= 1, 1 - np.abs((A - B) / bws), 0.0))

def c_biweight(A, B, bws, Cmatrix_unused):
    z2 = ((A - B) / bws) ** 2
    return np.sum(np.where(np.abs((A - B) / bws) <= 1, (15/32.0) * (3 - z2) ** 2, 0.0))

def c_triweight(A, B, bws, Cmatrix_unused):
    return np.sum(np.where(np.abs((A - B) / bws) <= 1, (35/32.0) * (1 - ((A - B) / bws) ** 2) ** 3, 0.0))

def c_tricube(A, B, bws, Cmatrix_unused):
    z = np.abs((A - B) / bws)
    return np.sum(np.where(z <= 1, (70/81.0) * (1 - z ** 3) ** 3, 0.0))

def c_cosine(A, B, bws, Cmatrix_unused):
    z = (A - B) / bws
    return np.sum(np.where(np.abs(z) <= 1, (np.pi/4.0) * np.cos((np.pi/2.0) * z), 0.0))

def c_logistic(A, B, bws, Cmatrix_unused):
    z = (A - B) / bws
    return np.sum(1.0 / (np.exp(z) + 2.0 + np.exp(-z)))

def c_sigmoid(A, B, bws, Cmatrix_unused):
    z = (A - B) / bws
    return np.sum(2.0 / (np.pi * (np.exp(z) + np.exp(-z))))

def c_silverman(A, B, bws, Cmatrix_unused):
    z = np.abs((A - B) / bws) / np.sqrt(2.0)
    return np.sum((1.0 / bws) * 0.5 * np.exp(-z) * np.sin(z + np.pi / 4.0))

# Unordered categorical kernels
def u_aitchisonaitken(A, B, bws, card_subset):
    """
    Aitchison–Aitken nominal kernel using correct per-feature category counts.
    card_subset must be an array of same length as A/B/bws giving K_j for each feature.
    """
    res = 0.0
    m = len(A)
    for j in range(m):
        K = int(card_subset[j]) if card_subset is not None else 2
        if K <= 1:
            res += (1 - bws[j])
        else:
            res += (1 - bws[j]) if A[j] == B[j] else bws[j] / (K - 1)
    return res


def u_aitken(A, B, bws, Cfull):
    res = 0.0
    m = len(A)
    for j in range(m):
        res += 1.0 if A[j] == B[j] else bws[j]
    return res

# Ordinal kernels
def o_wangvanryzin(A, B, bws, Cfull):
    if np.all(A == B):
        return float(np.sum(1 - bws))
    else:
        return float(np.sum(0.5 * (1 - bws) * (bws ** np.abs(A - B))))

def o_aitchisonaitken(A, B, bws, Cfull):
    return float(np.sum(np.where(A == B, 1.0, bws ** np.abs(A - B))))

def o_aitken(A, B, bws, Cfull):
    return float(np.sum(np.where(A == B, bws, (1 - bws) / (2 ** np.abs(A - B)))))

def o_habbema(A, B, bws, Cfull):
    return float(np.sum(bws ** (np.abs(A - B) ** 2)))

def o_liracine(A, B, bws, Cfull):
    return float(np.sum(np.where(A == B, 1.0, bws ** np.abs(A - B))))

# -------------------------
# KDSS distance (row vs center)
# -------------------------
def _select_kernel(_unused=None):
    ck = {
        "c_gaussian": c_gaussian, "c_uniform": c_uniform, "c_epanechnikov": c_epanechnikov,
        "c_triangle": c_triangle, "c_biweight": c_biweight, "c_triweight": c_triweight,
        "c_tricube": c_tricube, "c_cosine": c_cosine, "c_logistic": c_logistic,
        "c_sigmoid": c_sigmoid, "c_silverman": c_silverman
    }
    uk = {"u_aitken": u_aitken, "u_aitchisonaitken": u_aitchisonaitken}
    ok = {"o_wangvanryzin": o_wangvanryzin, "o_habbema": o_habbema,
          "o_aitken": o_aitken, "o_aitchisonaitken": o_aitchisonaitken, "o_liracine": o_liracine}
    return ck, uk, ok

def dkss_distance(x, c, bw,
                  cFUN="c_gaussian", uFUN="u_aitken", oFUN="o_wangvanryzin",
                  con_idx=None, fac_idx=None, ord_idx=None):
    x = np.asarray(x); c = np.asarray(c); bw = np.asarray(bw)
    assert x.shape == c.shape == bw.shape, "x, c, bw must be same length"

    if con_idx is None: con_idx = list(range(len(x)))
    if fac_idx is None: fac_idx = []
    if ord_idx is None: ord_idx = []

    ck, uk, ok = _select_kernel()
    c_fun = ck[cFUN]; u_fun = uk[uFUN]; o_fun = ok[oFUN]

    def compute(FUN, idx, kind):
        if len(idx) == 0: return 0.0
        a = x[idx]; b = c[idx]; bws = bw[idx]
        # For nominal features, pass the correct per-feature cardinalities (K_j) sliced to idx
        aux = None
        if kind == "nominal":
            # CAT_CARD is a length-D vector; slice it to the idx subset
            aux = CAT_CARD[idx] if CAT_CARD is not None else None
        # Continuous and ordinal kernels ignore the 4th arg
        return FUN(a, a, bws, aux) + FUN(b, b, bws, aux) - FUN(a, b, bws, aux) - FUN(b, a, bws, aux)

    return float(
        compute(c_fun, con_idx, "continuous")
      + compute(u_fun, fac_idx, "nominal")
      + compute(o_fun, ord_idx, "ordinal")
    )


# -------------------------
# FCM inner loop (no bw update) used inside GA objective
# -------------------------
def kdml_fcm_no_bw_update(X, C, bw, m, epsilon, con_idx, fac_idx, ord_idx, max_iter=1):
    N, D = X.shape
    U = np.random.rand(N, C)
    U /= U.sum(axis=1, keepdims=True)

    J_prev = float('inf')
    for _ in range(max_iter):
        # update centers (simple weighted mean on encoded features)
        V = np.zeros((C, D))
        for c in range(C):
            w = (U[:, c] ** m).reshape(-1, 1)
            num = np.sum(w * X, axis=0)
            den = np.sum(U[:, c] ** m)
            V[c] = num / max(den, 1e-12)

        # distances
        Dmat = np.zeros((N, C))
        for i in range(N):
            for c in range(C):
                Dmat[i, c] = dkss_distance(
                    X[i], V[c], bw=bw,
                    cFUN="c_gaussian", uFUN="u_aitken", oFUN="o_wangvanryzin",
                    con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx
                )

        # update U
        for i in range(N):
            for c in range(C):
                if Dmat[i, c] == 0:
                    U[i, :] = 0; U[i, c] = 1
                else:
                    denom = 0.0
                    for j in range(C):
                        denom += (Dmat[i, c] / (Dmat[i, j] if Dmat[i, j] > 0 else 1e-12)) ** (2.0 / (m - 1))
                    U[i, c] = 1.0 / max(denom, 1e-12)

        J = np.sum((U ** m) * (Dmat ** 2))
        if abs(J - J_prev) < epsilon: break
        J_prev = J

    return V, U

# -------------------------
# GA objective (negated FARI vs init membership)
# -------------------------
def objective(bw_candidate, U_ref, X, C, m, con_idx, fac_idx, ord_idx, epsilon=0.01, max_iter_obj=1):
    _, U = kdml_fcm_no_bw_update(X=X, C=C, bw=np.asarray(bw_candidate), m=m, epsilon=epsilon,
                                 con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx, max_iter=max_iter_obj)
    return -fari(U, U_ref)

# -------------------------
# GA components
# -------------------------
def compute_diversity(population):
    arr = np.array(population)
    return float(np.mean(np.std(arr, axis=0)))

def clamp_continuous(candidate, con_idx):
    cand = list(candidate)
    if con_idx is None:
        for i in range(len(cand)):
            cand[i] = max(cand[i], 0.1)
    else:
        for i in con_idx:
            cand[i] = max(cand[i], 0.1)
    return tuple(cand)

def tournament_selection(pop, tournament_size, U_ref, X, C, m, con_idx, fac_idx, ord_idx, epsilon=0.01, max_iter_obj=1):
    t = random.sample(pop, tournament_size)
    return min(t, key=lambda ind: objective(ind, U_ref=U_ref, X=X, C=C, m=m,
                                            con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx,
                                            epsilon=epsilon, max_iter_obj=max_iter_obj))

def one_point_crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def mutate(individual, p_m, sigma, con_idx, fac_idx, ord_idx):
    out = []
    for i, gene in enumerate(individual):
        g = gene
        if random.random() < p_m:
            g = g + np.random.normal(0, sigma)
            if (fac_idx and i in fac_idx) or (ord_idx and i in ord_idx):
                g = max(0.0, min(g, 1.0))
            else:
                g = max(g, 0.1)
        out.append(g)
    return tuple(out)

def genetic_algorithm(pop_size, num_generations, bandwidth_length, p_m, p_c, tournament_size, sigma,
                      U_ref, X, C, m, con_idx, fac_idx, ord_idx,
                      verbose=False, initial_bw=None, max_iter_obj=1, show_progress=True):
    from tqdm import tqdm
    
    population = []
    for _ in range(pop_size):
        cand = []
        for i in range(bandwidth_length):
            if (fac_idx and i in fac_idx) or (ord_idx and i in ord_idx):
                cand.append(random.uniform(0.0, 1.0))
            else:
                cand.append(random.uniform(0.001, 10.0))
        population.append(clamp_continuous(tuple(cand), con_idx))
    if initial_bw is not None:
        population[0] = clamp_continuous(tuple(initial_bw), con_idx)

    best = None
    best_fit = float('inf')
    prev_gen_best = float('inf')

    # Create progress bar for GA generations
    if show_progress:
        ga_pbar = tqdm(range(num_generations), desc="Genetic Algorithm", unit="gen", 
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    else:
        ga_pbar = range(num_generations)

    for gen in ga_pbar:
        gen_best_fit = float('inf'); gen_best = None
        fits = []
        for ind in population:
            f = objective(ind, U_ref=U_ref, X=X, C=C, m=m,
                          con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx,
                          max_iter_obj=max_iter_obj)
            fits.append(f)
            if f < gen_best_fit:
                gen_best_fit, gen_best = f, ind
            if f < best_fit:
                best_fit, best = f, ind

        # Update progress bar with current metrics
        if show_progress:
            div = compute_diversity(population)
            imp = 0 if prev_gen_best==float('inf') else (prev_gen_best - gen_best_fit)
            ga_pbar.set_postfix({
                'Best': f'{gen_best_fit:.4f}',
                'Avg': f'{np.mean(fits):.4f}',
                'Div': f'{div:.3f}',
                'Imp': f'{imp:.4f}'
            })
        
        if verbose:
            div = compute_diversity(population)
            imp = 0 if prev_gen_best==float('inf') else (prev_gen_best - gen_best_fit)
            print(f"[GA] gen={gen+1}/{num_generations} best={gen_best_fit:.4f} avg={np.mean(fits):.4f} div={div:.4f} imp={imp:.4f}")
        prev_gen_best = gen_best_fit

        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, tournament_size, U_ref, X, C, m, con_idx, fac_idx, ord_idx, max_iter_obj=max_iter_obj)
            p2 = tournament_selection(population, tournament_size, U_ref, X, C, m, con_idx, fac_idx, ord_idx, max_iter_obj=max_iter_obj)
            if random.random() < p_c:
                c1, c2 = one_point_crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            c1 = mutate(c1, p_m, sigma, con_idx, fac_idx, ord_idx)
            c2 = mutate(c2, p_m, sigma, con_idx, fac_idx, ord_idx)
            new_pop.extend([clamp_continuous(c1, con_idx), clamp_continuous(c2, con_idx)])
        population = new_pop[:pop_size]

    if show_progress:
        ga_pbar.close()

    return best, best_fit, population

# -------------------------
# DDQN agent (device-aware)
# -------------------------
VHC="VHC"; HC="HC"; LC="LC"; STALLED="Stalled"; INCREASED="Increased"
VHD="VHD"; HD="HD"; MD="MD"; LD="LD"; VLD="VLD"

REWARD_TABLE = {
    (VHC, VHD): 200,   (VHC, HD): 150,   (VHC, MD): 100,   (VHC, LD): 50,    (VHC, VLD): 25,
    (HC,  VHD): 150,   (HC,  HD): 112.5, (HC,  MD): 75,    (HC,  LD): 37.5,  (HC,  VLD): 18.75,
    (LC,  VHD): 100,   (LC,  HD): 75,    (LC,  MD): 50,    (LC,  LD): 25,    (LC,  VLD): 12.5,
    (STALLED, VHD): 0, (STALLED, HD): 0, (STALLED, MD): -10, (STALLED, LD): -20, (STALLED, VLD): -30,
    (INCREASED, VHD): -100, (INCREASED, HD): -150, (INCREASED, MD): -200, (INCREASED, LD): -250, (INCREASED, VLD): -300
}

def discretize_improvement(improvement):
    if improvement > 0.005: return VHC
    if improvement > 0.003: return HC
    if improvement > 0.001: return LC
    if improvement == 0:    return STALLED
    return INCREASED

def discretize_diversity(diversity):
    if diversity > 0.8: return VHD
    if diversity > 0.6: return HD
    if diversity > 0.4: return MD
    if diversity > 0.2: return LD
    return VLD

def state_to_index(state):
    imp_map = {VHC:0, HC:1, LC:2, STALLED:3, INCREASED:4}
    div_map = {VHD:0, HD:1, MD:2, LD:3, VLD:4}
    return imp_map[state[0]]*5 + div_map[state[1]]

def onehot_from_index(idx, size=25):
    v = np.zeros(size, dtype=np.float32); v[idx]=1; return v

def index_to_action(idx):
    p_m = (idx // 5)/4.0
    p_c = (idx % 5)/4.0
    return (p_m, p_c)

class QNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers=[]; prev=in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class DDQNAgent:
    def __init__(self, state_size=25, action_size=25, hidden_dims=[64,64],
                 lr=1e-3, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=32, target_update_freq=100, device=None):
        self.state_size=state_size; self.action_size=action_size
        self.gamma=gamma
        self.epsilon=epsilon; self.epsilon_min=epsilon_min; self.epsilon_decay=epsilon_decay
        self.batch_size=batch_size; self.target_update_freq=target_update_freq
        self.update_counter=0
        self.global_step=0  # Add global step counter for metrics
        self.device = device or pick_device()

        self.online_net = QNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net = QNetwork(state_size, hidden_dims, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict()); self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay = deque(maxlen=buffer_size)

        # Debug: confirm model device
        print(f"[DDQN] Using device: {self.device} | param device: {next(self.online_net.parameters()).device}")

    def get_state(self, improvement, diversity):
        return (discretize_improvement(improvement), discretize_diversity(diversity))

    def _encode(self, state):
        return onehot_from_index(state_to_index(state), self.state_size)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size-1)
        s = torch.from_numpy(self._encode(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(s)
        return int(q.argmax(dim=1))

    def store(self, s,a,r,s2,done):
        self.replay.append((s,a,r,s2,done))

    def update_network(self, metrics_logger=None):
        if len(self.replay) < self.batch_size: return
        batch = random.sample(self.replay, self.batch_size)
        S, A, R, S2, D = zip(*batch)
        S  = torch.from_numpy(np.vstack([self._encode(s)  for s  in S ])).to(self.device)
        S2 = torch.from_numpy(np.vstack([self._encode(s2) for s2 in S2])).to(self.device)
        A  = torch.tensor(A, dtype=torch.long, device=self.device).unsqueeze(1)
        R  = torch.tensor(R, dtype=torch.float32, device=self.device).unsqueeze(1)
        D  = torch.tensor(D, dtype=torch.float32, device=self.device).unsqueeze(1)

        Q  = self.online_net(S).gather(1, A)
        next_a = self.online_net(S2).argmax(dim=1, keepdim=True)
        Q2 = self.target_net(S2).gather(1, next_a)
        target = R + self.gamma * Q2 * (1 - D)

        loss = nn.MSELoss()(Q, target.detach())
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

        # Log TD loss for metrics
        if metrics_logger:
            metrics_logger.log_td_loss(self.global_step, loss.item())

        self.update_counter += 1
        self.global_step += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

    def step(self, s, a, r, s2, done=False, metrics_logger=None):
        self.store(s,a,r,s2,done)
        self.update_network(metrics_logger)

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.online_net.to(self.device); self.target_net.to(self.device)

    def assert_on_device(self):
        dev = self.device
        ok = all(p.device == dev for p in self.online_net.parameters())
        print(f"[DDQN] Params on {dev}: {ok}")
        return ok

# -------------------------
# RL wrapper around GA
# -------------------------
def estimate_bandwidth_vector_RL(state, initial_bw, RL_agent, prev_ga_obj,
                                 U_ref, X, C, m, con_idx, fac_idx, ord_idx,
                                 max_iter_obj=1, verbose=False, metrics_logger=None):
    r_list = []

    action_idx = RL_agent.choose_action(state)
    p_m, p_c = index_to_action(action_idx)
    if verbose:
        print(f"[DDQN] state={state} -> action p_m={p_m:.2f}, p_c={p_c:.2f}")

    # Log action for metrics
    if metrics_logger:
        metrics_logger.log_action(action_idx, state)

    best_bw, best_obj, final_pop = genetic_algorithm(
        pop_size=20, num_generations=10, bandwidth_length=len(initial_bw),
        p_m=p_m, p_c=p_c, tournament_size=3, sigma=5.0,
        U_ref=U_ref, X=X, C=C, m=m, con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx,
        verbose=False, initial_bw=initial_bw, max_iter_obj=max_iter_obj, show_progress=False
    )

    diversity = compute_diversity(final_pop)
    improvement = 0.0 if prev_ga_obj is None else (prev_ga_obj - best_obj)
    next_state = (discretize_improvement(improvement), discretize_diversity(diversity))
    reward = REWARD_TABLE[next_state]

    RL_agent.step(state, action_idx, reward, next_state, done=False, metrics_logger=metrics_logger)
    r_list.append(reward)

    if verbose:
        print(f"[DDQN] improvement={improvement:.6f} diversity={diversity:.4f} reward={reward}")

    return best_bw, best_obj, r_list, next_state

def train_RL_agent(X, initial_bw, RL_agent, prev_ga_obj, U_ref, C, m,
                   con_idx, fac_idx, ord_idx, B=200, max_iter_obj=1,
                   ckpt_dir="./checkpoints", verbose=False, show_progress=True):
    from tqdm import tqdm
    
    ensure_dir(ckpt_dir)
        # Initialize categorical cardinalities for nominal/ordinal kernels
    set_categorical_cardinalities(X, fac_idx=fac_idx, ord_idx=ord_idx)

    # Initialize metrics logger
    metrics_logger = MetricsLogger(out_dir=os.path.join(ckpt_dir, "rl_metrics"))
    
    rewards_all = []
    state = (STALLED, VLD)
    bw_current = list(initial_bw)

    # Create progress bar
    if show_progress:
        pbar = tqdm(range(B), desc="Training DDQN", unit="episode", 
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    else:
        pbar = range(B)

    for b in pbar:
        bw_current, ga_obj, r_list, state = estimate_bandwidth_vector_RL(
            state=state, initial_bw=bw_current, RL_agent=RL_agent, prev_ga_obj=prev_ga_obj,
            U_ref=U_ref, X=X, C=C, m=m, con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx,
            max_iter_obj=max_iter_obj, verbose=verbose, metrics_logger=metrics_logger
        )
        rewards_all.extend(r_list)
        prev_ga_obj = ga_obj

        # Log episode metrics
        episode_reward = sum(r_list) if r_list else 0.0
        metrics_logger.log_episode_reward(episode_reward)
        metrics_logger.log_epsilon(b, RL_agent.epsilon)
        metrics_logger.log_state(state)
        
        # Log Q-values and replay buffer stats periodically
        if b % 10 == 0:  # Every 10 episodes
            metrics_logger.log_q_values(RL_agent.global_step, RL_agent)
            metrics_logger.log_replay(RL_agent)

        # Update progress bar with current metrics
        if show_progress and len(rewards_all) > 0:
            avg_reward = np.mean(rewards_all[-10:])  # Last 10 rewards
            cum_reward = np.sum(rewards_all)
            pbar.set_postfix({
                'Epsilon': f'{RL_agent.epsilon:.3f}',
                'Avg Reward': f'{avg_reward:.2f}',
                'Cum Reward': f'{cum_reward:.1f}'
            })

        if (b+1) % 50 == 0:
            RL_agent.save(os.path.join(ckpt_dir, "ddqn_agent.pt"))

    if show_progress:
        pbar.close()

    RL_agent.save(os.path.join(ckpt_dir, "ddqn_agent.pt"))
    
    # Generate all RL metrics plots
    print(f"\n=== Generating RL Training Metrics ===")
    metrics_logger.plot_all()
    print(f"RL metrics plots saved to: {metrics_logger.out_dir}")

    # Plot cumulative reward
    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(rewards_all), marker='o')
    plt.title("Cumulative Reward during DDQN Training")
    plt.xlabel("Episode"); plt.ylabel("Cumulative Reward"); plt.grid(True)
    ensure_dir("./outputs"); plt.savefig("./outputs/ddqn_training_rewards.png", dpi=200, bbox_inches="tight"); plt.close()

    return RL_agent, rewards_all

# -------------------------
# Full KDSS-FCM with RL-GA bandwidth search
# -------------------------
def centers_from_membership(X, U, m):
    C = U.shape[1]; D = X.shape[1]
    V = np.zeros((C, D))
    for c in range(C):
        w = (U[:, c] ** m).reshape(-1, 1)
        num = np.sum(w * X, axis=0)
        den = np.sum(U[:, c] ** m)
        V[c] = num / max(den, 1e-12)
    return V

def kdml_fcm(X, C, m, epsilon, con_idx, fac_idx, ord_idx,
             U_init, max_iter=100, verbose=False,
             RL_agent=None, train_B=0, max_iter_obj=1, ckpt_dir="./checkpoints", show_fcm_progress=True):
    N, D = X.shape

    # Ensure lists
    con_idx = [] if con_idx is None else list(con_idx)
    fac_idx = [] if fac_idx is None else list(fac_idx)
    ord_idx = [] if ord_idx is None else list(ord_idx)

    # Precompute cardinalities for nominal/ordinal kernels
    set_categorical_cardinalities(X, fac_idx=fac_idx, ord_idx=ord_idx)


    # Initialize fuzzy membership from provided init membership (normalize)
    U = np.asarray(U_init, dtype=float)
    U = U / np.clip(U.sum(axis=1, keepdims=True), 1e-12, None)
    # Track clustering history per-iteration
    U_history = [U.copy()]

    # Initialize centers from that membership
    V = centers_from_membership(X, U, m)

    # Random start bandwidths, aligned to original columns
    bw_current = make_aligned_bandwidths(D, con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx)



    # Train / init agent
    if RL_agent is None:
        RL_agent = DDQNAgent(lr=1e-3, gamma=0.9, epsilon=1.0)

    if train_B and train_B > 0:
        RL_agent, _ = train_RL_agent(
            X=X, initial_bw=bw_current, RL_agent=RL_agent, prev_ga_obj=None,
            U_ref=U, C=C, m=m, con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx,
            B=train_B, max_iter_obj=max_iter_obj, ckpt_dir=ckpt_dir, verbose=verbose
        )

    # RL loop over FCM iterations
    J_prev = float('inf')
    reward_history = []
    state = RL_agent.get_state(0.0, 1.0)  # start state

    # Create progress bar for FCM iterations (only if enabled)
    from tqdm import tqdm
    if show_fcm_progress:
        fcm_pbar = tqdm(range(max_iter), desc="KDSS-FCM", unit="iter", 
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    else:
        fcm_pbar = range(max_iter)

    prev_ga_obj = None
    for it in fcm_pbar:
        if verbose:
            print(f"\n[FCM] iter={it}")

        # One RL-GA step to update bandwidth
        bw_current, ga_obj, r_list, state = estimate_bandwidth_vector_RL(
            state=state, initial_bw=bw_current, RL_agent=RL_agent, prev_ga_obj=prev_ga_obj,
            U_ref=U, X=X, C=C, m=m, con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx,
            max_iter_obj=max_iter_obj, verbose=verbose
        )
        prev_ga_obj = ga_obj
        reward_history.extend(r_list)

        # Update centers from current U
        V = centers_from_membership(X, U, m)

        # Distances with new bw
        Dmat = np.zeros((N, C))
        for i in range(N):
            for c in range(C):
                Dmat[i, c] = dkss_distance(
                    X[i], V[c], np.asarray(bw_current),
                    cFUN="c_gaussian", uFUN="u_aitken", oFUN="o_wangvanryzin",
                    con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx
                )

        # Update U
        for i in range(N):
            for c in range(C):
                if Dmat[i, c] == 0:
                    U[i, :] = 0; U[i, c] = 1
                else:
                    denom = 0.0
                    for j in range(C):
                        denom += (Dmat[i, c] / (Dmat[i, j] if Dmat[i, j] > 0 else 1e-12)) ** (2.0 / (m - 1))
                    U[i, c] = 1.0 / max(denom, 1e-12)

        J = np.sum((U ** m) * (Dmat ** 2))
        # record clustering for this iteration
        U_history.append(U.copy())
        
        # Update progress bar with current metrics (only if progress bar is active)
        if show_fcm_progress:
            fcm_pbar.set_postfix({
                'J': f'{J:.4f}',
                'ΔJ': f'{abs(J-J_prev):.2e}',
                'Reward': f'{np.sum(reward_history):.1f}'
            })
        
        if verbose:
            print(f"[FCM] J={J:.6f}  Δ={abs(J-J_prev):.3e}  bw[0..3]={np.asarray(bw_current)[:3]}")
        if abs(J - J_prev) < epsilon:
            if verbose: print(f"[FCM] Converged at iter {it}")
            if show_fcm_progress:
                fcm_pbar.set_postfix({'Status': 'Converged'})
            break
        J_prev = J

    if show_fcm_progress:
        fcm_pbar.close()

    # Plot cumulative reward
    ensure_dir("./outputs")
    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(reward_history))
    plt.title("Cumulative Reward during KDSS-FCM (DDQN-controlled)")
    plt.xlabel("Iteration"); plt.ylabel("Cumulative Reward"); plt.grid(True)
    plt.savefig("./outputs/fcm_cumulative_reward.png", dpi=200, bbox_inches="tight"); plt.close()

    return V, U, reward_history, np.asarray(bw_current), U_history
