"""Is all-honest (all-D) a Nash profile? Unilateral deviations of every node, constant attack cost.

Utility (normalised by a3 - a2 > 0), discounted over attacked rounds s+1..T, constant cost:
    U_i = sum_t w_t [rho q_i^t - qbar_i^t] - c_A 1[s_i > 0]
Node i deviating to intensity s (message l - s (l - w)) while everyone else is honest gains
    G_i(s) = sum_t w_t [rho dq_i - dqbar_i]   (paired vs all-D, same seed),
so all-D is Nash iff c_A >= c_A*_i = max_s G_i(s) for every node i.

Two-stage to avoid selection bias (8 intensities x 16 nodes):
  search seeds  -> per node and rho, select s_i = argmax mean gain (any sign)
  holdout seeds -> estimate G_i(s_i): mean, and upper bound mean + T_CRIT SE
Deterrence claims use the holdout upper bound; the search-seed maximum is reported but biased up.
Both quality definitions (local validation q, balanced audit) come from the same simulations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import b4_nash as b
from dfl_lambda import DEFENSES, lattice

RHOS = (1.0, 1.25, 1.5, 2.0, 3.0)


def gains(outcome, clean, i, nb, rho, w):
    """[seed] gain of node i: outcome/clean are [seed, K, N] quality arrays."""
    d = outcome - clean
    return (rho * d[:, :, i] - d[:, :, list(nb[i])].mean(2)) @ w


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--defense", choices=DEFENSES, default="uniform")
    p.add_argument("--output", type=Path)
    p.add_argument("--cache", default=".cache/mnist")
    p.add_argument("--search-seeds", type=int, nargs="+", default=[7, 19, 31, 43, 59])
    p.add_argument("--holdout-seeds", type=int, nargs="+", default=[61, 67, 71, 73, 79])
    p.add_argument("--intensities", type=float, nargs="+", default=[2, 3, 4, 5, 6, 8, 12, 16])
    p.add_argument("--discount", type=float, default=0.95)
    p.add_argument("--workers", type=int, default=min(22, os.cpu_count() or 1))
    # Simulator and defense settings (defaults = values used in B1-B6).
    for k, v in (("nodes", 16), ("alpha", 1.0), ("rounds", 40), ("attack_start", 10), ("samples", 1024),
                 ("validation", 200), ("audit_per_class", 100), ("hidden", 64), ("batch_size", 64),
                 ("lr", 0.1), ("gate_tol", 1.0), ("trim_b", 1), ("balance_gamma", 0.3), ("balance_kappa", 1.0),
                 ("balance_alpha", 0.5), ("flag_tol", 0.5), ("trust_beta", 0.5), ("trust_thr", 0.75),
                 ("trust_warmup", 5), ("punish_rounds", 0), ("grace_rounds", 0)):
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    p.add_argument("--mode", default="non_iid")
    p.add_argument("--val-dist", choices=["local", "balanced"], default="local")
    args = p.parse_args()
    suffix = "" if args.val_dist == "local" else f"_val{args.val_dist}"
    out = args.output or Path(f"experiments/results/alld_{args.defense}{suffix}_lattice_noniid.json")
    sim = {k: getattr(args, k) for k in ("nodes", "mode", "alpha", "rounds", "attack_start", "samples",
                                         "validation", "audit_per_class", "hidden", "batch_size", "lr", "defense",
                                         "gate_tol", "trim_b", "balance_gamma", "balance_kappa", "balance_alpha",
                                         "flag_tol", "trust_beta", "trust_thr", "trust_warmup", "val_dist",
                                         "punish_rounds", "grace_rounds")}
    n, nb = args.nodes, lattice(args.nodes)
    K = args.rounds - args.attack_start
    w = args.discount ** np.arange(K)
    w /= w.sum()
    seeds = args.search_seeds + args.holdout_seeds
    honest = (0.0,) * n
    profiles = [honest] + [tuple(s if j == i else 0.0 for j in range(n)) for i in range(n) for s in args.intensities]
    start = time.time()
    with Pool(args.workers, initializer=b._init, initargs=(args.cache, sim)) as pool:
        jobs = [(sd, pr, True) for pr in profiles for sd in seeds]
        res = pool.map(b.simulate, jobs, chunksize=4)
    elapsed = time.time() - start
    Q = {"q": {}, "audit": {}}
    for (sd, pr, _), (q, au) in zip(jobs, res):
        Q["q"][(sd, pr)] = q
        Q["audit"][(sd, pr)] = au

    def arr(metric, pr, ss):
        return np.stack([Q[metric][(sd, pr)] for sd in ss])

    report = {"config": {**sim, "defense": args.defense, "discount": args.discount, "intensities": args.intensities,
                         "search_seeds": args.search_seeds, "holdout_seeds": args.holdout_seeds},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "elapsed_s": round(elapsed, 1), "results": {}}
    print(f"{args.defense}: {len(jobs)} simulations in {elapsed:.0f}s")
    for metric in ("q", "audit"):
        cs, ch = arr(metric, honest, args.search_seeds), arr(metric, honest, args.holdout_seeds)
        tail_h = 100 * ch[:, -10:].mean()
        print(f"\n== metric {metric}: clean holdout quality (last 10 rounds) {tail_h:.2f}; gains x100 ==")
        print("  rho  | c_A* = max_i holdout UCB (node, s)   | nodes with holdout-significant gain>0 | median node UCB")
        for rho in RHOS:
            rows = []
            for i in range(n):
                prof = {s: tuple(s if j == i else 0.0 for j in range(n)) for s in args.intensities}
                g_search = {s: gains(arr(metric, pr, args.search_seeds), cs, i, nb, rho, w) for s, pr in prof.items()}
                s_sel = max(args.intensities, key=lambda s: g_search[s].mean())
                g_hold = gains(arr(metric, prof[s_sel], args.holdout_seeds), ch, i, nb, rho, w)
                mean, se = g_hold.mean(), g_hold.std(ddof=1) / np.sqrt(len(g_hold))
                rows.append({"node": i, "s": s_sel, "search_mean": float(g_search[s_sel].mean()),
                             "holdout_mean": float(mean), "holdout_se": float(se),
                             "holdout_ucb": float(mean + b.T_CRIT * se),
                             "holdout_significant": bool(mean > b.T_CRIT * se),
                             "search_curve": {str(s): float(g.mean()) for s, g in g_search.items()}})
            report["results"][f"{metric}_rho{rho}"] = rows
            worst = max(rows, key=lambda r: r["holdout_ucb"])
            sig = [r["node"] for r in rows if r["holdout_significant"]]
            print(f"  {rho:4.2f} | {100 * max(worst['holdout_ucb'], 0):7.3f} (node {worst['node']:2d}, s={worst['s']:g})"
                  f"{'':13s}| {len(sig):2d}/{n} {sig if sig else ''}"
                  f"{'':4s}| {100 * np.median([r['holdout_ucb'] for r in rows]):7.3f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
