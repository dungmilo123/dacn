"""Tables and figures for B1 (clean DFL vs local-only) and B2 (damage curve D(lambda)).

All B2 quantities are paired differences against the clean DFL run of the same seed
(identical split, init and minibatch order), sign convention: positive = quality lost.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfl_lambda import hops


def ms(values):
    v = np.asarray(values, float)
    return f"{100 * v.mean():6.2f} ± {100 * v.std(ddof=1):5.2f}" if len(v) > 1 else f"{100 * v.mean():6.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("experiments/results/b1_b2_lattice_noniid.json"))
    args = p.parse_args()
    d = json.loads(args.input.read_text())
    cfg, nb = d["config"], d["neighbors"]
    T, s, n = cfg["rounds"], cfg["attack_start"], cfg["nodes"]
    seeds = cfg["seeds"]
    runs = {(r["seed"], r["case"], r["attacker"], r["lam"]): r for r in d["runs"]}
    Q = {k: np.asarray(r["q"]) for k, r in runs.items()}
    A = {k: np.asarray(r["audit"]) for k, r in runs.items()}
    qbar = lambda q: np.stack([q[:, list(nb[i])].mean(1) for i in range(n)], 1)
    out = args.input.with_suffix("")
    tail = slice(T - 9, T + 1)  # last 10 evaluations

    print("== B1: clean, lattice 4x4, non-IID alpha=%s, %d seeds, T=%d ==" % (cfg["alpha"], len(seeds), T))
    print("mean over nodes, averaged over last 10 rounds; %% mean ± sd over seeds")
    for case in ("local", "dfl"):
        q = [Q[(sd, case, None, 0.0)][tail].mean() for sd in seeds]
        a = [A[(sd, case, None, 0.0)][tail].mean() for sd in seeds]
        qmin = [Q[(sd, case, None, 0.0)][tail].mean(0).min() for sd in seeds]
        amin = [A[(sd, case, None, 0.0)][tail].mean(0).min() for sd in seeds]
        print(f"  {case:5s}  q(local-val) {ms(q)}  min-node {ms(qmin)} | audit(balanced) {ms(a)}  min-node {ms(amin)}")
    gq = np.concatenate([Q[(sd, "dfl", None, 0.0)][tail].mean(0) - Q[(sd, "local", None, 0.0)][tail].mean(0) for sd in seeds])
    ga = np.concatenate([A[(sd, "dfl", None, 0.0)][tail].mean(0) - A[(sd, "local", None, 0.0)][tail].mean(0) for sd in seeds])
    print(f"  node-level DFL gain: q>0 in {np.sum(gq > 0)}/{len(gq)} node-seeds (mean {100*gq.mean():+.2f}), "
          f"audit>0 in {np.sum(ga > 0)}/{len(ga)} (mean {100*ga.mean():+.2f})")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for case, style in (("local", "--"), ("dfl", "-")):
        for k, (src, name) in enumerate(((Q, "q_i: local validation"), (A, "balanced audit"))):
            m = np.mean([src[(sd, case, None, 0.0)].mean(1) for sd in seeds], 0)
            ax[k].plot(m, style, label=case)
            ax[k].set_title(name); ax[k].set_xlabel("round"); ax[k].set_ylim(0, 1)
    ax[0].legend(); fig.tight_layout(); fig.savefig(f"{out}_b1.png", dpi=120)

    attackers = sorted({k[2] for k in runs if k[2] is not None})
    lams = sorted({k[3] for k in runs if k[2] is not None})
    print("\n== B2: one attacker from round %d, uniform aggregation; damage = clean - attacked (pp) ==" % s)
    print("D1   = drop of attacker's neighbour mean qbar at the first affected round (one-shot)")
    print("Dss  = same, averaged over all attacked rounds")
    print("own  = attacker's own q drop averaged over attacked rounds (feedback through aggregation)")
    print("net  = drop of balanced-audit accuracy of honest nodes, last 10 rounds")
    curves = {}
    for att in attackers:
        h = np.array(hops(nb, att))
        honest = np.arange(n) != att
        print(f"\n attacker {att} (degree {len(nb[att])})")
        print("   lambda |      D1        |      Dss       |      own       |      net       | hop1 Dss | hop2 Dss | hop3+ Dss")
        for lam in lams:
            r = {k: [] for k in ("D1", "Dss", "own", "net", "h1", "h2", "h3")}
            for sd in seeds:
                qc, qa = Q[(sd, "dfl", None, 0.0)], Q[(sd, "dfl", att, lam)]
                ac, aa = A[(sd, "dfl", None, 0.0)], A[(sd, "dfl", att, lam)]
                dqbar = qbar(qc)[:, att] - qbar(qa)[:, att]
                dq = qc - qa
                r["D1"].append(dqbar[s + 1]); r["Dss"].append(dqbar[s + 1:].mean())
                r["own"].append(dq[s + 1:, att].mean())
                r["net"].append((ac - aa)[tail][:, honest].mean())
                r["h1"].append(dq[s + 1:, h == 1].mean()); r["h2"].append(dq[s + 1:, h == 2].mean())
                r["h3"].append(dq[s + 1:, h >= 3].mean())
            curves[(att, lam)] = {k: np.asarray(v) for k, v in r.items()}
            print(f"   {lam:6.1f} | {ms(r['D1'])} | {ms(r['Dss'])} | {ms(r['own'])} | {ms(r['net'])} |"
                  f" {100*np.mean(r['h1']):8.2f} | {100*np.mean(r['h2']):8.2f} | {100*np.mean(r['h3']):8.2f}")
        small = [l for l in lams if l <= 1]
        D1 = np.array([curves[(att, l)]["D1"].mean() for l in small])
        beta = float(np.dot(small, D1) / np.dot(small, small))
        print(f"   beta (one-shot slope through origin, lambda<=1): {beta:.4f} per unit lambda")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for att in attackers:
        for k, key in enumerate(("D1", "Dss", "own")):
            m = [100 * curves[(att, l)][key].mean() for l in lams]
            e = [100 * curves[(att, l)][key].std(ddof=1) for l in lams]
            ax[k].errorbar([0, *lams], [0, *m], yerr=[0, *e], marker="o", capsize=3,
                           label=f"attacker {att} (deg {len(nb[att])})")
    for k, title in enumerate(("D1: one-shot neighbour damage", "Dss: sustained neighbour damage",
                               "own: attacker's own q drop")):
        ax[k].set_xscale("symlog", linthresh=1); ax[k].set_xlabel("lambda"); ax[k].set_ylabel("pp lost")
        ax[k].set_title(title); ax[k].axhline(0, color="gray", lw=0.5)
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out}_b2.png", dpi=120)
    print(f"\nfigures: {out}_b1.png, {out}_b2.png")


if __name__ == "__main__":
    main()
