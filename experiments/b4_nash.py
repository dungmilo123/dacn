"""B4: every node strategic; search a Nash profile (lambda_1..lambda_N) by iterated best response.

Normalised utility (divide u_i by a3 - a2 > 0), per attacked round t = s+1..T, discount d:
    U_i = sum_t d^t [rho q_i^t - qbar_i^t] / sum_t d^t - c lambda_i^2
    rho = (a1 + a3)/(a3 - a2),  c = g k / (a3 - a2)
Each node holds a stationary lambda_i from round s (same restricted policy class as B3).
Every profile is simulated on all seeds (paired: same split, init, minibatch order).
Alternatives that beat a node's current lambda by more than T_CRIT paired standard errors
over seeds are "significant"; the node moves to the significant one with the highest mean.
A sweep with no switch is a statistical Nash profile on the lambda grid (no significant
profitable unilateral deviation). Nodes update one at a time (Gauss-Seidel), order 0..N-1.

Rounds 0..s are clean and identical for every profile, so each worker caches the round-s
snapshot per seed. Reuses the B2 simulator primitives; official test never loaded.
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

from dfl_lambda import DEFENSES, accuracy, init_trust, initialize, lattice, load_mnist, make_split, step

T_CRIT = 2.776  # one-sided 97.5% Student t, 4 df (5 paired seeds)
_X = _Y = None
_ARGS = None
_SNAP: dict = {}


def _init(cache, args):
    global _X, _Y, _ARGS
    _X, _Y = load_mnist(Path(cache))
    _ARGS = argparse.Namespace(**args)


def _snapshot(seed):
    """Round-s state (split, served models, trust) of the clean run; identical for every profile."""
    if seed not in _SNAP:
        a, nb = _ARGS, lattice(_ARGS.nodes)
        split = make_split(_Y, seed, a)
        w = [initialize(seed, a.hidden) for _ in range(a.nodes)]
        trust = init_trust(a.nodes, a)
        zero = [0.0] * a.nodes
        for t in range(a.attack_start):
            w, trust, _, _ = step(w, trust, t, seed, zero, split, nb, a, _X, _Y)
        _SNAP[seed] = (split, w, trust)
    return _SNAP[seed]


def simulate(job):
    """job = (seed, profile tuple, with_audit). Returns q (and audit) for rounds s+1..T, shape [K, N]."""
    seed, profile, with_audit = job
    a, x, y, nb = _ARGS, _X, _Y, lattice(_ARGS.nodes)
    split, w, trust = _snapshot(seed)
    _, val_idx, audit_idx = split
    q, au = [], []
    for t in range(a.attack_start, a.rounds):
        w, trust, _, _ = step(w, trust, t, seed, list(profile), split, nb, a, x, y)
        q.append([accuracy(w[i], x[val_idx[i]], y[val_idx[i]]) for i in range(a.nodes)])
        if with_audit:
            au.append([accuracy(w[i], x[audit_idx], y[audit_idx]) for i in range(a.nodes)])
    return np.asarray(q), (np.asarray(au) if with_audit else None)


class Game:
    def __init__(self, pool, seeds, metric, nb):
        self.pool, self.seeds, self.metric, self.nb = pool, seeds, metric, nb
        self.cache: dict = {}
        self.sims = 0

    def outcomes(self, profiles):
        """Simulate any uncached (seed, profile); returns {profile: array [seed, K, N]} of the game metric."""
        audit = self.metric == "audit"
        todo = [(sd, p, audit) for p in profiles for sd in self.seeds if (sd, p) not in self.cache]
        for (sd, p, _), (q, au) in zip(todo, self.pool.map(simulate, todo)):
            self.cache[(sd, p)] = au if audit else q
        self.sims += len(todo)
        return {p: np.stack([self.cache[(sd, p)] for sd in self.seeds]) for p in profiles}

    def utility(self, q, i, lam, rho, c, w):
        """[seed] normalised discounted utility of node i."""
        qbar = q[:, :, list(self.nb[i])].mean(2)
        return (rho * q[:, :, i] - qbar) @ w - c * lam ** 2


def best_response_dynamics(game, grid, rho, c, w, nodes, max_sweeps):
    profile = (0.0,) * nodes
    history = [list(profile)]
    for sweep in range(max_sweeps):
        changed = 0
        for i in range(nodes):
            options = [tuple(l if j == i else profile[j] for j in range(nodes)) for l in grid]
            out = game.outcomes(options)
            U = {l: game.utility(out[p], i, l, rho, c, w) for l, p in zip(grid, options)}
            cur = profile[i]
            better = []
            for l in grid:
                diff = U[l] - U[cur]
                if l != cur and diff.mean() > T_CRIT * diff.std(ddof=1) / np.sqrt(len(diff)):
                    better.append(l)
            if better:
                profile = options[grid.index(max(better, key=lambda l: U[l].mean()))]
                changed += 1
        history.append(list(profile))
        print(f"    sweep {sweep + 1}: {changed} switches, profile {[f'{l:g}' for l in profile]}", flush=True)
        if not changed:
            return profile, history, True
    return profile, history, False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=".cache/mnist")
    p.add_argument("--output", type=Path, default=Path("experiments/results/b4_lattice_noniid.json"))
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31, 43, 59])
    p.add_argument("--rhos", type=float, nargs="+", default=[1.25, 1.5, 2.0])
    p.add_argument("--costs", type=float, nargs="+", default=[1e-4, 1e-3])
    p.add_argument("--metric", choices=["q", "audit"], default="q")
    p.add_argument("--grid", type=float, nargs="+", default=[0, 4, 5, 6, 8, 10, 12])
    p.add_argument("--discount", type=float, default=0.95)
    p.add_argument("--max-sweeps", type=int, default=6)
    p.add_argument("--nodes", type=int, default=16)
    p.add_argument("--mode", default="non_iid")
    p.add_argument("--val-dist", choices=["local", "balanced"], default="local")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--attack-start", type=int, default=10)
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--validation", type=int, default=200)
    p.add_argument("--audit-per-class", type=int, default=100)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--defense", choices=DEFENSES, default="uniform")
    p.add_argument("--gate-tol", type=float, default=1.0)
    p.add_argument("--trim-b", type=int, default=1)
    p.add_argument("--balance-gamma", type=float, default=0.3)
    p.add_argument("--balance-kappa", type=float, default=1.0)
    p.add_argument("--balance-alpha", type=float, default=0.5)
    p.add_argument("--flag-tol", type=float, default=0.5)
    p.add_argument("--trust-beta", type=float, default=0.5)
    p.add_argument("--trust-thr", type=float, default=0.75)
    p.add_argument("--trust-warmup", type=int, default=5)
    p.add_argument("--punish-rounds", type=int, default=0)
    p.add_argument("--grace-rounds", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    args = p.parse_args()
    sim_args = {k: getattr(args, k) for k in ("nodes", "mode", "alpha", "rounds", "attack_start", "samples",
                                              "validation", "audit_per_class", "hidden", "batch_size", "lr",
                                              "defense", "gate_tol", "trim_b", "balance_gamma",
                                              "balance_kappa", "balance_alpha", "flag_tol", "trust_beta",
                                              "trust_thr", "trust_warmup", "val_dist", "punish_rounds",
                                              "grace_rounds")}
    nb = lattice(args.nodes)
    K = args.rounds - args.attack_start
    w = args.discount ** np.arange(K)
    w /= w.sum()
    grid = [float(g) for g in args.grid]
    start = time.time()
    results = []
    with Pool(args.workers, initializer=_init, initargs=(args.cache, sim_args)) as pool:
        game = Game(pool, args.seeds, args.metric, nb)
        honest = (0.0,) * args.nodes
        for rho in args.rhos:
            for c in args.costs:
                print(f"== rho={rho} c={c:g} metric={args.metric} ==", flush=True)
                profile, history, converged = best_response_dynamics(game, grid, rho, c, w, args.nodes, args.max_sweeps)
                # Report both quality views at the final profile and at all-honest.
                rep = pool.map(simulate, [(sd, pr, True) for pr in (honest, profile) for sd in args.seeds])
                n = len(args.seeds)
                def welfare(qs):
                    return float(np.mean([sum(game.utility(q[None], i, profile_l, rho, c, w)[0]
                                              for i, profile_l in enumerate(pr)) / args.nodes
                                          for q, pr in qs]))
                metric_idx = 0 if args.metric == "q" else 1
                res = {
                    "rho": rho, "c": c, "converged": converged, "profile": list(profile), "history": history,
                    "attackers": int(sum(l > 0 for l in profile)),
                    "q_tail_honest": float(np.mean([r[0][-10:].mean() for r in rep[:n]])),
                    "q_tail_nash": float(np.mean([r[0][-10:].mean() for r in rep[n:]])),
                    "audit_tail_honest": float(np.mean([r[1][-10:].mean() for r in rep[:n]])),
                    "audit_tail_nash": float(np.mean([r[1][-10:].mean() for r in rep[n:]])),
                    "welfare_honest": welfare([(r[metric_idx], honest) for r in rep[:n]]),
                    "welfare_nash": welfare([(r[metric_idx], profile) for r in rep[n:]]),
                }
                results.append(res)
                print(f"    -> {res['attackers']} attackers; q {100*res['q_tail_honest']:.2f} -> {100*res['q_tail_nash']:.2f};"
                      f" audit {100*res['audit_tail_honest']:.2f} -> {100*res['audit_tail_nash']:.2f};"
                      f" welfare {res['welfare_honest']:.4f} -> {res['welfare_nash']:.4f}"
                      f" | {game.sims} sims, {time.time() - start:.0f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "neighbors": nb, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_s": round(time.time() - start, 1), "results": results}))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
