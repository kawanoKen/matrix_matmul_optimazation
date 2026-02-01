import subprocess
import csv
import io
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import math
import random
import time
import numpy as np
from run_matmul import Config, run_matmul, make_candidates
import matplotlib.pyplot as plt


def random_search(cands: List[Config], budget: int, seed: int = 0, **run_kwargs):
    rng = random.Random(seed)
    best_cfg, best_val = None, float("inf")
    history = []

    for it in range(budget):
        cfg = rng.choice(cands)
        val = run_matmul(cfg, **run_kwargs)
        history.append({
            "iter": it,
            "cfg": cfg,
            "time": val,
        })
        if val < best_val:
            best_cfg, best_val = cfg, val
            print(f"[RS] it={it:03d} best={best_val:.3f} ms cfg={best_cfg}")
        else:
            print(f"[RS] it={it:03d} val={val:.3f} ms best={best_val:.3f}")
    return best_cfg, best_val, history


def featurize(cfg: Config) -> np.ndarray:
    return np.array([
        cfg.BM / 128.0,
        cfg.BN / 128.0,
        cfg.BK / 32.0,
        math.log2(cfg.U) / 3.0,   # U in {1,2,4,8} -> {0..3}/3
        cfg.T / 8.0,
    ], dtype=np.float64)

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, lengthscale: float = 0.6, var: float = 1.0):
    X1_sq = np.sum(X1**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2**2, axis=1, keepdims=True).T
    dist2 = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)
    return var * np.exp(-0.5 * dist2 / (lengthscale**2))

def gp_fit_predict(X_train, y_train, X_test, noise: float = 1e-6,
                   lengthscale: float = 0.6, var: float = 1.0):

    K = rbf_kernel(X_train, X_train, lengthscale, var)
    K[np.diag_indices_from(K)] += noise
    Ks = rbf_kernel(X_train, X_test, lengthscale, var)   # (n, m)
    Kss = rbf_kernel(X_test, X_test, lengthscale, var)   # (m, m)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    mu = Ks.T @ alpha  # (m,)

    v = np.linalg.solve(L, Ks)
    cov = Kss - v.T @ v
    var_pred = np.maximum(np.diag(cov), 1e-12)
    std = np.sqrt(var_pred)
    return mu, std


def normal_pdf(z):
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)

def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def expected_improvement_min(mu, sigma, best_y, xi=0.01):
    if sigma < 1e-12:
        return 0.0
    imp = best_y - mu - xi
    z = imp / sigma
    return imp * normal_cdf(z) + sigma * normal_pdf(z)

def ego_gp_ei(cands: List[Config], budget: int, n_init: int = 10, seed: int = 0, **run_kwargs):
    rng = random.Random(seed)
    X_all = np.stack([featurize(c) for c in cands], axis=0)

    idx_all = list(range(len(cands)))
    rng.shuffle(idx_all)
    init_idx = idx_all[:n_init]

    X_train, y_train, seen = [], [], set()
    best_cfg, best_val = None, float("inf")

    for it, idx in enumerate(init_idx):
        cfg = cands[idx]
        val = run_matmul(cfg, **run_kwargs)
        seen.add(idx)
        X_train.append(X_all[idx])
        y_train.append(val)
        if val < best_val:
            best_cfg, best_val = cfg, val
        print(f"[EGO-init] it={it:03d} val={val:.3f} ms best={best_val:.3f} cfg={cfg}")

    history = []
    for it in range(n_init, budget):
        Xtr = np.array(X_train, dtype=np.float64)
        ytr = np.array(y_train, dtype=np.float64)

        cand_idx = [i for i in range(len(cands)) if i not in seen]
        Xte = X_all[cand_idx]

        mu, std = gp_fit_predict(Xtr, ytr, Xte, noise=1e-6, lengthscale=0.6, var=1.0)

        eis = [expected_improvement_min(mu[i], std[i], best_val, xi=0.01) for i in range(len(cand_idx))]
        pick = cand_idx[int(np.argmax(eis))]

        cfg = cands[pick]
        val = run_matmul(cfg, **run_kwargs)
        seen.add(pick)

        X_train.append(X_all[pick])
        y_train.append(val)
        history.append({
            "iter": it,
            "cfg": cfg,
            "time": val,
        })

        if val < best_val:
            best_cfg, best_val = cfg, val
            print(f"[EGO] it={it:03d} NEW BEST={best_val:.3f} ms cfg={best_cfg}")
        else:
            print(f"[EGO] it={it:03d} val={val:.3f} ms best={best_val:.3f}")

    return best_cfg, best_val, history

def best_so_far(history):
    best = float("inf")
    xs, ys = [], []
    for h in history:
        best = min(best, h["time"])
        xs.append(h["iter"])
        ys.append(best)
    return xs, ys

def plot_convergence(rs_hist, ego_hist, save_path=None):
    x_rs, y_rs = best_so_far(rs_hist)
    x_ego, y_ego = best_so_far(ego_hist)

    plt.figure(figsize=(7, 5))
    plt.plot(x_rs, y_rs, label="Random Search", marker="o")
    plt.plot(x_ego, y_ego, label="EGO (GP + EI)", marker="s")

    plt.xlabel("Number of evaluations")
    plt.ylabel("Best execution time (ms)")
    plt.title("Convergence of optimization methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    # plt.show()

def plot_search_distribution(history, title, save_path=None):
    BMs = [h["cfg"].BM for h in history]
    BNs = [h["cfg"].BN for h in history]
    times = [h["time"] for h in history]

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(BMs, BNs, c=times, cmap="viridis", s=80, edgecolor="k")
    plt.colorbar(sc, label="Execution time (ms)")

    plt.xlabel("BM (tile size M)")
    plt.ylabel("BN (tile size N)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    # plt.show()




def main():
    cands = make_candidates()

    run_kwargs = dict(M=1024, N=1024, K=1024, repeat=5, exe="./matmul")

    budget = 40

    print("\n=== Random Search ===")
    rs_best_cfg, rs_best_val, rs_hist = random_search(cands, budget=budget, seed=1234, **run_kwargs)
    print("[RS] BEST:", rs_best_cfg, rs_best_val, "ms")

    print("\n=== EGO (GP + EI) ===")
    ego_best_cfg, ego_best_val, ego_hist = ego_gp_ei(cands, budget=budget, n_init=10, seed=1234, **run_kwargs)
    print("[EGO] BEST:", ego_best_cfg, ego_best_val, "ms")

    plot_convergence(rs_hist, ego_hist, save_path="convergence.png")
    plot_search_distribution(rs_hist, "Random Search", save_path="search_distribution_rs.png")
    plot_search_distribution(ego_hist, "EGO", save_path="search_distribution_ego.png")

if __name__ == "__main__":
    main()
