# rl_metrics.py
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ----- helpers to index your 5x5 discrete state grid -----
_IMP_ORDER = ["VHC", "HC", "LC", "Stalled", "Increased"]
_DIV_ORDER = ["VHD", "HD", "MD", "LD", "VLD"]
_IMP_TO_I = {s:i for i,s in enumerate(_IMP_ORDER)}
_DIV_TO_J = {s:j for j,s in enumerate(_DIV_ORDER)}

def state_rc(state_tuple):
    # state is like ("HC", "MD")
    return _IMP_TO_I[state_tuple[0]], _DIV_TO_J[state_tuple[1]]

def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

class MetricsLogger:
    """
    Tracks & plots the 8 recommended RL training metrics:
      1) per-episode reward (smoothed)
      2) evaluation return (no exploration)
      3) epsilon schedule
      4) TD loss (moving avg)
      5) state visitation heatmap (5x5)
      6) action usage (hist + optional state-conditioned)
      7) Q-value sanity (avg max-Q online/target)
      8) replay buffer stats (size and mean age)
    """
    def __init__(self, out_dir="./outputs/rl_training", smooth_window=50):
        self.out_dir = out_dir
        ensure_dir(out_dir)

        # (1) rewards per episode
        self.episode_rewards = []

        # (2) evaluation return
        self.eval_steps = []
        self.eval_returns = []

        # (3) epsilon per episode
        self.eps_episodes = []
        self.eps_values = []

        # (4) TD loss per update
        self.td_steps = []
        self.td_losses = []

        # (5) state visitation
        self.state_counts = np.zeros((5,5), dtype=np.int64)  # [imp_row, div_col]

        # (6) action usage
        self.action_counts = np.zeros(25, dtype=np.int64)
        # optional: actions conditioned on state (25 per 5x5)
        self.action_counts_by_state = np.zeros((5,5,25), dtype=np.int64)

        # (7) Q-values sanity
        self.q_steps = []
        self.avg_max_q_online = []
        self.avg_max_q_target = []

        # (8) replay buffer stats
        self.replay_steps = []
        self.replay_sizes = []
        self.replay_mean_age = []

        self.smooth_window = smooth_window

    # --------- logging APIs ----------
    def log_episode_reward(self, reward_sum):
        self.episode_rewards.append(float(reward_sum))

    def log_epsilon(self, episode_idx, epsilon):
        self.eps_episodes.append(int(episode_idx))
        self.eps_values.append(float(epsilon))

    def log_td_loss(self, step_idx, loss_value):
        if loss_value is None or (isinstance(loss_value, float) and math.isnan(loss_value)):
            return
        self.td_steps.append(int(step_idx))
        self.td_losses.append(float(loss_value))

    def log_state(self, state_tuple):
        r, c = state_rc(state_tuple)
        self.state_counts[r, c] += 1

    def log_action(self, action_idx, state_tuple=None):
        self.action_counts[int(action_idx)] += 1
        if state_tuple is not None:
            r, c = state_rc(state_tuple)
            self.action_counts_by_state[r, c, int(action_idx)] += 1

    def log_eval_return(self, global_step, eval_return):
        self.eval_steps.append(int(global_step))
        self.eval_returns.append(float(eval_return))

    def log_q_values(self, global_step, agent, eval_all_25_states=True, custom_states=None):
        """
        Compute avg max-Q online/target for either:
          - all 25 one-hot states (default), or
          - a provided list of state tuples.
        """
        import torch
        states = []
        if eval_all_25_states:
            for i_imp in range(5):
                for j_div in range(5):
                    st = (_IMP_ORDER[i_imp], _DIV_ORDER[j_div])
                    states.append(st)
        elif custom_states:
            states = list(custom_states)
        else:
            return  # nothing to do

        with torch.no_grad():
            X = []
            for st in states:
                idx = agent._encode(st)  # one-hot (25,)
                X.append(idx)
            X = np.stack(X, axis=0)
            X_t = torch.from_numpy(X).to(agent.device)

            q_on = agent.online_net(X_t)
            q_tg = agent.target_net(X_t)

            self.q_steps.append(int(global_step))
            self.avg_max_q_online.append(float(q_on.max(dim=1).values.mean().item()))
            self.avg_max_q_target.append(float(q_tg.max(dim=1).values.mean().item()))

    def log_replay(self, agent):
        """
        Assumes agent.replay stores tuples (..., step_idx) as last element.
        Mean age = current_step - stored_step averaged over sampled buffer.
        """
        size = len(agent.replay)
        if size == 0:
            return
        cur = int(getattr(agent, "global_step", 0))
        ages = []
        try:
            # tuples are (s,a,r,s2,done,step)
            ages = [cur - t[-1] for t in agent.replay if isinstance(t[-1], int)]
        except Exception:
            ages = []
        mean_age = float(np.mean(ages)) if ages else 0.0

        self.replay_steps.append(cur)
        self.replay_sizes.append(size)
        self.replay_mean_age.append(mean_age)

    # ---------- plotting ----------
    def _rolling(self, x, w):
        if len(x) == 0:
            return np.array([])
        w = max(1, min(w, len(x)))
        c = np.cumsum(np.insert(x, 0, 0.0))
        out = (c[w:] - c[:-w]) / w
        pad = np.full(w-1, np.nan)
        return np.concatenate([pad, out])

    def plot_1_reward(self):
        y = np.array(self.episode_rewards, dtype=float)
        y_sm = self._rolling(y, self.smooth_window)
        plt.figure(figsize=(8,5))
        plt.plot(y, alpha=0.35, linewidth=1, label="Episode reward")
        plt.plot(y_sm, linewidth=2, label=f"Rolling mean (w={self.smooth_window})")
        # shaded band: rolling std (simple, centered on mean)
        if len(y) >= self.smooth_window:
            stds = []
            w = self.smooth_window
            for i in range(len(y)):
                lo = max(0, i-w+1); hi = i+1
                stds.append(np.std(y[lo:hi]))
            stds = np.array(stds)
            plt.fill_between(np.arange(len(y)), y_sm - stds, y_sm + stds, alpha=0.15, label="±1 SD")
        plt.title("Per-episode Reward (smoothed)")
        plt.xlabel("Episode"); plt.ylabel("Reward")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(self.out_dir, "01_episode_reward.png"), dpi=200); plt.close()

    def plot_2_eval_return(self):
        if not self.eval_steps:
            return
        plt.figure(figsize=(8,5))
        plt.plot(self.eval_steps, self.eval_returns, marker='o')
        plt.title("Evaluation Return (ε=0)")
        plt.xlabel("Global step"); plt.ylabel("Eval return")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(self.out_dir, "02_eval_return.png"), dpi=200); plt.close()

    def plot_3_epsilon(self):
        if not self.eps_episodes:
            return
        plt.figure(figsize=(8,5))
        plt.plot(self.eps_episodes, self.eps_values)
        plt.title("Epsilon vs Episode")
        plt.xlabel("Episode"); plt.ylabel("ε")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(self.out_dir, "03_epsilon.png"), dpi=200); plt.close()

    def plot_4_td_loss(self):
        if not self.td_steps:
            return
        y = np.array(self.td_losses, dtype=float)
        # simple EMA for smoothing
        ema = []
        beta = 0.99
        m = 0.0
        for v in y:
            m = beta*m + (1.0-beta)*v
            ema.append(m/(1.0 - beta))
        plt.figure(figsize=(8,5))
        plt.plot(self.td_steps, y, alpha=0.3, linewidth=1, label="TD loss")
        plt.plot(self.td_steps, ema, linewidth=2, label="EMA(β=0.99)")
        plt.title("TD Loss over Updates")
        plt.xlabel("Update step"); plt.ylabel("MSE(Q, target)")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(self.out_dir, "04_td_loss.png"), dpi=200); plt.close()

    def plot_5_state_heatmap(self):
        if self.state_counts.sum() == 0:
            return
        mat = self.state_counts.astype(float)
        total = mat.sum()
        pct = (mat / max(1.0, total)) * 100.0

        plt.figure(figsize=(7,6))
        # draw heatmap manually (no seaborn)
        for i in range(5):
            for j in range(5):
                v = pct[i,j]
                plt.gca().add_patch(plt.Rectangle((j, 4-i), 1, 1, fill=True, alpha=0.5))
                plt.text(j+0.5, 4-i+0.5, f"{v:.1f}%", ha="center", va="center", fontsize=10)
        plt.xlim(0,5); plt.ylim(0,5)
        plt.xticks(np.arange(5)+0.5, _DIV_ORDER)
        plt.yticks(np.arange(5)+0.5, list(reversed(_IMP_ORDER)))
        plt.title("State Visitation Heatmap (% of visits)")
        plt.grid(True, which='both', linewidth=0.5)
        plt.tight_layout(); plt.savefig(os.path.join(self.out_dir, "05_state_visitation.png"), dpi=200); plt.close()

    def plot_6_action_usage(self):
        if self.action_counts.sum() == 0:
            return
        plt.figure(figsize=(9,5))
        plt.bar(np.arange(25), self.action_counts)
        plt.title("Action Usage (counts)")
        plt.xlabel("Action index (0..24)"); plt.ylabel("Count")
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(self.out_dir, "06_action_usage_hist.png"), dpi=200); plt.close()

        # optional: state-conditioned usage for a couple of representative states
        # (uncomment to save a small grid of bar plots)
        # for (ri, rn) in enumerate(_IMP_ORDER):
        #     for (cj, cn) in enumerate(_DIV_ORDER):
        #         counts = self.action_counts_by_state[ri, cj]
        #         if counts.sum() == 0: continue
        #         plt.figure(figsize=(9,5))
        #         plt.bar(np.arange(25), counts)
        #         plt.title(f"Action Usage | state=({rn},{cn})")
        #         plt.xlabel("Action index (0..24)"); plt.ylabel("Count")
        #         plt.grid(True, axis='y', alpha=0.3)
        #         fn = f"06_action_usage_state_{rn}_{cn}.png".replace("/", "-")
        #         plt.tight_layout(); plt.savefig(os.path.join(self.out_dir, fn), dpi=200); plt.close()

    def plot_7_q_sanity(self):
        if not self.q_steps:
            return
        plt.figure(figsize=(8,5))
        plt.plot(self.q_steps, self.avg_max_q_online, label="Online: avg max-Q")
        plt.plot(self.q_steps, self.avg_max_q_target, label="Target: avg max-Q", linestyle="--")
        plt.title("Q-value Sanity (all 25 states)")
        plt.xlabel("Global step"); plt.ylabel("Average max-Q")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(self.out_dir, "07_q_sanity.png"), dpi=200); plt.close()

    def plot_8_replay_stats(self):
        if not self.replay_steps:
            return
        fig, ax1 = plt.subplots(figsize=(8,5))
        ax1.plot(self.replay_steps, self.replay_sizes, label="Buffer size")
        ax1.set_xlabel("Global step"); ax1.set_ylabel("Buffer size")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(self.replay_steps, self.replay_mean_age, label="Mean age", linestyle="--")
        ax2.set_ylabel("Mean transition age (steps)")
        fig.suptitle("Replay Buffer Stats")
        fig.tight_layout(); fig.savefig(os.path.join(self.out_dir, "08_replay_stats.png"), dpi=200); plt.close(fig)

    def plot_all(self):
        self.plot_1_reward()
        self.plot_2_eval_return()
        self.plot_3_epsilon()
        self.plot_4_td_loss()
        self.plot_5_state_heatmap()
        self.plot_6_action_usage()
        self.plot_7_q_sanity()
        self.plot_8_replay_stats()
