"""B7 screening of W variants under constant attack cost (one attacker at corner 0 / edge 1 / interior 5).

Per variant (one dfl_lambda.py output):
  clean    : quality of all-honest, last 10 rounds (q = node validation, audit = balanced audit)
  damage   : neighbour qbar drop and attacker's own q drop, discounted mean over attacked rounds, at s=2,3,4
  c_A*     : max_s of the repeated-game gain G(s) = sum_t w_t [dqbar - rho dq_own], keeping only s whose
             gain is significant over seeds (analyze_b3.significant_mean); '-' = no significant gain.
             All-D resists this attacker position iff the constant attack cost c_A >= c_A*.
Screening only (search seeds, 3 positions); deterrence claims need alld_check.py with holdout seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_b3 import curves, significant_mean

RHOS = (1.0, 1.25, 1.5, 2.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="label=path pairs")
    p.add_argument("--discount", type=float, default=0.95)
    args = p.parse_args()
    data = {}
    for item in args.inputs:
        label, path = item.split("=", 1)
        data[label] = json.loads(Path(path).read_text())
    any_d = next(iter(data.values()))
    T, s0 = any_d["config"]["rounds"], any_d["config"]["attack_start"]
    K = T - s0
    w = args.discount ** np.arange(K)
    w /= w.sum()

    print("== clean (all-D), last 10 rounds, % mean over seeds ==")
    for label, d in data.items():
        clean = [r for r in d["runs"] if r["case"] == "dfl" and r["attacker"] is None]
        q = np.mean([np.asarray(r["q"])[-10:].mean() for r in clean])
        a = np.mean([np.asarray(r["audit"])[-10:].mean() for r in clean])
        amin = np.mean([np.asarray(r["audit"])[-10:].mean(0).min() for r in clean])
        print(f"  {label:14s} q {100 * q:6.2f} | audit {100 * a:6.2f} | audit min-node {100 * amin:6.2f}")

    for metric in ("q", "audit"):
        print(f"\n== metric {metric}: damage at s=2/3/4 (pp, discounted mean; attacker 5 / 1 / 0) ==")
        print(f"  {'variant':14s} | {'neighbour qbar drop':30s} | attacker own drop")
        cur_all = {label: curves(d, metric) for label, d in data.items()}
        for label, cur in cur_all.items():
            nb_cells, own_cells = [], []
            for s in (2.0, 3.0, 4.0):
                nb_cells.append("/".join(f"{100 * (cur[(a, s)]['D'] @ w).mean():.1f}" for a in (5, 1, 0)))
                own_cells.append("/".join(f"{100 * (cur[(a, s)]['Own'] @ w).mean():.1f}" for a in (5, 1, 0)))
            print(f"  {label:14s} | {'  '.join(nb_cells):30s} | {'  '.join(own_cells)}")
        print(f"\n== metric {metric}: constant-cost c_A* x100 (s at max) per attacker 5 / 1 / 0 ==")
        print(f"  {'variant':14s} | " + " | ".join(f"rho={r:<4g}{'':22s}" for r in RHOS))
        for label, cur in cur_all.items():
            lams = np.array(sorted({l for _, l in cur}))
            cells = []
            for rho in RHOS:
                parts = []
                for a in (5, 1, 0):
                    per_seed = np.stack([(cur[(a, l)]["D"] - rho * cur[(a, l)]["Own"]) @ w for l in lams], 1)
                    v = significant_mean(per_seed)
                    k = int(np.argmax(v))
                    parts.append(f"{100 * v[k]:.2f}({lams[k]:g})" if v[k] > 0 else "-")
                cells.append(" / ".join(f"{x:>9s}" for x in parts))
            print(f"  {label:14s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
