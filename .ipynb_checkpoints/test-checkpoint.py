#!/usr/bin/env python3
# rl_ga_kdss_fcm_infer.py
# Inference-only: load a trained DDQN agent, run KDSS-FCM+GA on a new dataset,
# save final FARI score and clustering plots as PNGs.
# Requires: numpy, pandas, torch, matplotlib, tqdm, scipy (scipy not strictly used)

import os
import sys
import math
import argparse
import random
from collections import deque
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D projection)

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
    Heuristic (only used if you don't pass --con_idx/--fac_idx/--ord_idx):
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
    Returns a numeric DataFrame (ints/floats).
    """
    df2 = df.copy()
    n_cols = df2.shape[1]

    fac_idx = [] if fac_idx is None else list(fac_idx)
    ord_idx = [] if ord_idx is None else list(ord_idx)

    # 1) Nominal (unordered)
    for j in fac_idx:
        s = df2.iloc[:, j]
        if not isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype("category")
        s = s.cat.as_unordered()
        df2.iloc[:, j] = s.cat.codes.astype("int64")

    # 2) Ordinal (ordered)
    for j in ord_idx:
        s = df2.iloc[:, j]
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.cat.as_ordered()
            df2.iloc[:, j] = pd.Series(s).cat.codes.astype("int64")
        else:
            try:
                vals = pd.to_numeric(s, errors="raise")
                cats = np.unique(vals.dropna())
                cat_strings = [str(c) for c in np.sort(cats)]
                s_str = s.astype(str)
                s_ord = pd.Categorical(s_str, categories=cat_strings, ordered=True)
                df2.iloc[:, j] = pd.Series(s_ord).cat.codes.astype("int64")
            except Exception:
                cats = sorted(s.astype(str).dropna().unique())
                s_ord = pd.Categorical(s.astype(str), categories=cats, ordered=True)
                df2.iloc[:, j] = pd.Series(s_ord).cat.codes.astype("int64")

    # 3) Remaining non-numeric -> nominal codes
    for j in range(n_cols):
        s = df2.iloc[:, j]
        if not pd.api.types.is_numeric_dtype(s):
            s = s.astype("category")
            df2.iloc[:, j] = s.cat.codes.astype("int64")

    # 4) Ensure numeric
    for col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    return df2

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
def u_aitchisonaitken(A, B, bws, Cfull):
    res = 0.0
    m = len(A)
    for j in range(m):
        unique_count = len(np.unique(Cfull[:, j]))
        if unique_count <= 1:
            res += (1 - bws[j])
        else:
            res += (1 - bws[j]) if A[j] == B[j] else bws[j] / (unique_count - 1)
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

    def compute(FUN, idx):
        if len(idx) == 0: return 0.0
        a = x[idx]; b = c[idx]; bws = bw[idx]
        full = np.vstack([x, c])[:, idx]  # small "full" matrix (2 x |idx|)
        return FUN(a, a, bws, full) + FUN(b, b, bws, full) - FUN(a, b, bws, full) - FUN(b, a, bws, full)

    return float(compute(c_fun, con_idx) + compute(u_fun, fac_idx) + compute(o_fun, ord_idx))

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
                      verbose=False, initial_bw=None, max_iter_obj=1):
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

    for gen in range(num_generations):
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

    return best, best_fit, population

# -------------------------
# DDQN agent (device-aware) - INFERENCE ONLY
# -------------------------
VHC="VHC"; HC="HC"; LC="LC"; STALLED="Stalled"; INCREASED="Increased"
VHD="VHD"; HD="HD"; MD="MD"; LD="LD"; VLD="VLD"

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
                 lr=1e-3, gamma=0.9, epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0,
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

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay = deque(maxlen=buffer_size)

        print(f"[DDQN] Using device: {self.device} | param device: {next(self.online_net.parameters()).device}")

    def get_state(self, improvement, diversity):
        return (discretize_improvement(improvement), discretize_diversity(diversity))

    def _encode(self, state):
        return onehot_from_index(state_to_index(state), self.state_size)

    def choose_action(self, state):
        # Inference only: greedy
        s = torch.from_numpy(self._encode(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(s)
        return int(q.argmax(dim=1))

    # No-op training hooks for inference
    def store(self, *args, **kwargs): pass
    def update_network(self, *args, **kwargs): pass
    def step(self, *args, **kwargs): pass

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
# RL wrapper around GA (Inference only: no agent updates)
# -------------------------
def estimate_bandwidth_vector_RL_infer(state, initial_bw, RL_agent,
                                       U_ref, X, C, m, con_idx, fac_idx, ord_idx,
                                       max_iter_obj=1, verbose=False):
    action_idx = RL_agent.choose_action(state)
    p_m, p_c = index_to_action(action_idx)
    if verbose:
        print(f"[DDQN] [INFER] state={state} -> action p_m={p_m:.2f}, p_c={p_c:.2f}")

    best_bw, best_obj, final_pop = genetic_algorithm(
        pop_size=20, num_generations=10, bandwidth_length=len(initial_bw),
        p_m=p_m, p_c=p_c, tournament_size=3, sigma=5.0,
        U_ref=U_ref, X=X, C=C, m=m, con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx,
        verbose=False, initial_bw=initial_bw, max_iter_obj=max_iter_obj
    )

    diversity = compute_diversity(final_pop)
    improvement = 0.0  # not tracked during inference
    next_state = (discretize_improvement(improvement), discretize_diversity(diversity))
    return best_bw, best_obj, next_state

# -------------------------
# Full KDSS-FCM with RL-GA bandwidth search (Inference-only)
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

def kdml_fcm_infer(X, C, m, epsilon, con_idx, fac_idx, ord_idx,
                   U_init, max_iter=100, verbose=False,
                   RL_agent=None, max_iter_obj=1):
    N, D = X.shape

    # Ensure lists
    con_idx = [] if con_idx is None else list(con_idx)
    fac_idx = [] if fac_idx is None else list(fac_idx)
    ord_idx = [] if ord_idx is None else list(ord_idx)

    # Initialize fuzzy membership from provided init membership (normalize)
    U = np.asarray(U_init, dtype=float)
    U = U / np.clip(U.sum(axis=1, keepdims=True), 1e-12, None)

    # Initialize centers from that membership
    V = centers_from_membership(X, U, m)

    # Random start bandwidths per feature type (fallback to one-per-col if mismatch)
    bw_current = [np.random.uniform(0.1,100) for _ in con_idx] \
               + [np.random.uniform(0,1)   for _ in fac_idx] \
               + [np.random.uniform(0,1)   for _ in ord_idx]
    if len(bw_current) != D:
        bw_current = [np.random.uniform(0.1, 10.0) for _ in range(D)]
        con_idx = list(range(D)); fac_idx=[]; ord_idx=[]

    if RL_agent is None:
        RL_agent = DDQNAgent(lr=1e-3, gamma=0.9, epsilon=0.0)  # greedy

    J_prev = float('inf')
    state = RL_agent.get_state(0.0, 1.0)  # start state

    for it in range(max_iter):
        if verbose:
            print(f"\n[FCM] iter={it}")

        # One RL-GA step to update bandwidth (no agent updates)
        bw_current, ga_obj, state = estimate_bandwidth_vector_RL_infer(
            state=state, initial_bw=bw_current, RL_agent=RL_agent,
            U_ref=U, X=X, C=C, m=m, con_idx=con_idx, fac_idx=fac_idx, ord_idx=ord_idx,
            max_iter_obj=max_iter_obj, verbose=verbose
        )

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
        if verbose:
            print(f"[FCM] J={J:.6f}  Δ={abs(J-J_prev):.3e}  bw[0..3]={np.asarray(bw_current)[:3]}")
        if abs(J - J_prev) < epsilon:
            if verbose: print(f"[FCM] Converged at iter {it}")
            break
        J_prev = J

    return V, U, np.asarray(bw_current)

# -------------------------
# Plotting helpers (no PCA)
# -------------------------
def labels_from_U(U):
    return U.argmax(axis=1)

def plot_2d_argmax(X, U, out_path, feature_names=None):
    labels = labels_from_U(U)
    plt.figure(figsize=(7,6))
    sc = plt.scatter(X[:,0], X[:,1], c=labels, s=20, cmap='tab10')
    plt.xlabel(feature_names[0] if feature_names else "x0")
    plt.ylabel(feature_names[1] if feature_names else "x1")
    plt.title("Clustering (argmax labels, 2D)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_2d_fuzzy_rgb(X, U, out_path, feature_names=None):
    C = U.shape[1]
    if C >= 3:
        rgb = U[:, :3]
        rgb = rgb / (rgb.max(axis=0, keepdims=True) + 1e-12)
    elif C == 2:
        rgb = np.zeros((U.shape[0], 3))
        rgb[:,0] = U[:,0]; rgb[:,1] = U[:,1]
    else:
        rgb = np.repeat(U, 3, axis=1)
    rgb = np.clip(rgb, 0, 1)
    plt.figure(figsize=(7,6))
    plt.scatter(X[:,0], X[:,1], c=rgb, s=20)
    plt.xlabel(feature_names[0] if feature_names else "x0")
    plt.ylabel(feature_names[1] if feature_names else "x1")
    plt.title("Fuzzy clustering (RGB memberships, 2D)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_3d_argmax(X, U, out_path, feature_names=None):
    labels = labels_from_U(U)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(X[:,0], X[:,1], X[:,2], c=labels, s=12, cmap='tab10', depthshade=True)
    ax.set_xlabel(feature_names[0] if feature_names else "x0")
    ax.set_ylabel(feature_names[1] if feature_names else "x1")
    ax.set_zlabel(feature_names[2] if feature_names else "x2")
    ax.set_title("Clustering (argmax labels, 3D)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_3d_fuzzy_rgb(X, U, out_path, feature_names=None):
    C = U.shape[1]
    if C >= 3:
        rgb = U[:, :3]
        rgb = rgb / (rgb.max(axis=0, keepdims=True) + 1e-12)
    elif C == 2:
        rgb = np.zeros((U.shape[0], 3))
        rgb[:,0] = U[:,0]; rgb[:,1] = U[:,1]
    else:
        rgb = np.repeat(U, 3, axis=1)
    rgb = np.clip(rgb, 0, 1)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=rgb, s=12, depthshade=True)
    ax.set_xlabel(feature_names[0] if feature_names else "x0")
    ax.set_ylabel(feature_names[1] if feature_names else "x1")
    ax.set_zlabel(feature_names[2] if feature_names else "x2")
    ax.set_title("Fuzzy clustering (RGB memberships, 3D)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_pairsplot_4d(X, U, out_path, feature_names=None):
    """Scatter matrix (4x4) colored by argmax labels; diagonal are histograms per cluster."""
    labels = labels_from_U(U)
    k = 4
    names = feature_names if (feature_names and len(feature_names) >= k) else [f"x{i}" for i in range(k)]
    uniq = np.unique(labels)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(int(u) % 10) for u in uniq]

    fig, axes = plt.subplots(k, k, figsize=(12,12))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if i == j:
                # overlaid histograms by cluster
                for u, col in zip(uniq, colors):
                    ax.hist(X[labels==u, j], bins=30, alpha=0.5, color=col)
                ax.set_ylabel("count")
            else:
                ax.scatter(X[:, j], X[:, i], c=labels, s=6, cmap='tab10')
            if i == k-1:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(names[i])
            else:
                ax.set_yticklabels([])
    fig.suptitle("Pairsplot (4D) by cluster labels", y=0.93)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_membership_heatmap(U, out_path):
    # Sort rows by argmax for readability
    order = np.argsort(U.argmax(axis=1))
    U_sorted = U[order]
    plt.figure(figsize=(8, max(4, U.shape[1])))
    plt.imshow(U_sorted.T, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Membership')
    plt.yticks(range(U.shape[1]), [f"Cluster {i}" for i in range(U.shape[1])])
    plt.xlabel("Samples (sorted by argmax)"); plt.ylabel("Clusters")
    plt.title("Membership matrix (heatmap)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Inference-only KDSS-FCM with pre-trained DDQN agent")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to data CSV")
    parser.add_argument("--init_membership_csv", type=str, required=True, help="Path to init membership CSV")
    parser.add_argument("--true_post_csv", type=str, required=True, help="Path to true posterior CSV")
    parser.add_argument("--agent_ckpt", type=str, required=True, help="Path to ddqn_agent.pt checkpoint")

    parser.add_argument(
        "--use_cols", type=str, default=None,
        help="Comma-separated 0-based column indices to use from the data CSV (e.g., '0,1,2'). "
             "If omitted, all columns are used."
    )
    parser.add_argument(
        "--con_idx", type=str, default=None,
        help="Comma-separated indices (relative to the USED columns) that are continuous."
    )
    parser.add_argument(
        "--fac_idx", type=str, default=None,
        help="Comma-separated indices (relative to the USED columns) that are NOMINAL (unordered categorical)."
    )
    parser.add_argument(
        "--ord_idx", type=str, default=None,
        help="Comma-separated indices (relative to the USED columns) that are ORDINAL (ordered categorical)."
    )

    parser.add_argument("--m", type=float, default=1.2)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--max_iter_obj", type=int, default=1, help="Inner FCM lookahead for GA objective")
    parser.add_argument("--outputs_dir", type=str, default="./outputs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.outputs_dir)

    # 1) Load CSVs
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
    n = X.shape[0]
    if U_init.shape[0] != n or U_true.shape[0] != n:
        raise ValueError(f"Row mismatch: data n={n}, init_U n={U_init.shape[0]}, true_post n={U_true.shape[0]}")
    C = U_init.shape[1]
    print(f"Inferred clusters C = {C}")

    # 5) Load DDQN agent on device (greedy)
    device = pick_device()
    print(f"[System] Selected device: {device}")
    agent = DDQNAgent(lr=1e-3, gamma=0.9, epsilon=0.0, device=device)  # epsilon=0.0 forces greedy
    agent.assert_on_device()
    agent.load(args.agent_ckpt)
    print(f"[Agent] Loaded from {args.agent_ckpt}")

    # Optional quick device check
    s_test = torch.from_numpy(agent._encode(agent.get_state(0.0, 1.0))).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_test = agent.online_net(s_test)
    print("[Check] Q tensor device:", q_test.device)

    # 6) Run KDSS-FCM (inference-only)
    V_final, U_final, bw_final = kdml_fcm_infer(
        X=X,
        C=C,
        m=args.m,
        epsilon=args.epsilon,
        con_idx=con_idx,
        fac_idx=fac_idx,
        ord_idx=ord_idx,
        U_init=U_init,
        max_iter=args.max_iters,
        verbose=args.verbose,
        RL_agent=agent,
        max_iter_obj=args.max_iter_obj
    )

    # 7) Evaluate and persist numeric outputs
    fari_vs_truth = fari(U_final, U_true)
    fari_vs_init  = fari(U_final, U_init)
    print("\n=== RESULTS (Inference) ===")
    print(f"FARI(U_final, true_post) = {fari_vs_truth:.6f}")
    print(f"FARI(U_final, init_U)    = {fari_vs_init:.6f}")
    print(f"Final bandwidth (all): {np.asarray(bw_final)}")

    np.savetxt(os.path.join(args.outputs_dir, "U_final.csv"), U_final, delimiter=",")
    np.savetxt(os.path.join(args.outputs_dir, "V_final.csv"), V_final, delimiter=",")
    np.savetxt(os.path.join(args.outputs_dir, "bw_final.csv"), np.asarray(bw_final), delimiter=",")
    with open(os.path.join(args.outputs_dir, "fari.txt"), "w") as f:
        f.write(f"FARI_vs_true={fari_vs_truth:.6f}\nFARI_vs_init={fari_vs_init:.6f}\n")
    print(f"Saved: {args.outputs_dir}/U_final.csv, V_final.csv, bw_final.csv, fari.txt")

    # 8) Plots (NO PCA):
    D = X.shape[1]
    # Use feature names of the used columns (helps axis labels)
    names = feature_names

    if D == 2:
        plot_2d_argmax(
            X, U_final,
            out_path=os.path.join(args.outputs_dir, "clustering_argmax_2d.png"),
            feature_names=names[:2]
        )
        plot_2d_fuzzy_rgb(
            X, U_final,
            out_path=os.path.join(args.outputs_dir, "clustering_fuzzy_rgb_2d.png"),
            feature_names=names[:2]
        )
        print(f"Saved plots: {args.outputs_dir}/clustering_argmax_2d.png, clustering_fuzzy_rgb_2d.png")

    elif D == 3:
        plot_3d_argmax(
            X, U_final,
            out_path=os.path.join(args.outputs_dir, "clustering_argmax_3d.png"),
            feature_names=names[:3]
        )
        plot_3d_fuzzy_rgb(
            X, U_final,
            out_path=os.path.join(args.outputs_dir, "clustering_fuzzy_rgb_3d.png"),
            feature_names=names[:3]
        )
        print(f"Saved plots: {args.outputs_dir}/clustering_argmax_3d.png, clustering_fuzzy_rgb_3d.png")

    elif D == 4:
        plot_pairsplot_4d(
            X, U_final,
            out_path=os.path.join(args.outputs_dir, "clustering_pairsplot_4d.png"),
            feature_names=names[:4]
        )
        print(f"Saved plot: {args.outputs_dir}/clustering_pairsplot_4d.png")

    elif D > 4:
        # Fallback: pairsplot of the first 4 features
        print(f"[Note] Data has {D} dimensions; generating a pairsplot for the first 4 features only.")
        plot_pairsplot_4d(
            X[:, :4], U_final,
            out_path=os.path.join(args.outputs_dir, "clustering_pairsplot_top4.png"),
            feature_names=names[:4]
        )
        print(f"Saved plot: {args.outputs_dir}/clustering_pairsplot_top4.png")

    # Membership heatmap (all cases)
    plot_membership_heatmap(
        U_final,
        out_path=os.path.join(args.outputs_dir, "membership_heatmap.png")
    )
    print(f"Saved plot: {args.outputs_dir}/membership_heatmap.png")

if __name__ == "__main__":
    main()
