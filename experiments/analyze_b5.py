"""B5: compare receive-side defenses on the B1/B2/B3 quantities.

Per defense (every node applies it; attacker included):
  clean   : B1 quality with no attacker, and its cost against uniform aggregation
  damage  : B2 paired drops (clean - attacked, same defense) of neighbour qbar / own q / honest audit
  c*      : B3 critical cost; attack is profitable iff c = g k/(a3-a2) < c* (significant gains only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_b3 import curves, significant_mean

RESULTS = Path("experiments/results")
FILES = {"uniform": RESULTS / "b1_b2_lattice_noniid.json",
         **{d: RESULTS / f"b5_{d}_lattice_noniid.json" for d in ("gate", "median", "trimmed", "clip", "balance")}}
RHOS = (1.0, 1.25, 1.5, 2.0)


def pm(v):
    v = np.asarray(v, float)
    return f"{100 * v.mean():6.2f} ± {100 * v.std(ddof=1):4.2f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--discount", type=float, default=0.95)
    p.add_argument("--metric", choices=["q", "audit"], default="q")
    args = p.parse_args()
    data = {d: json.loads(f.read_text()) for d, f in FILES.items() if f.exists()}
    nb = data["uniform"]["neighbors"]
    cfg = data["uniform"]["config"]
    T, s, n = cfg["rounds"], cfg["attack_start"], cfg["nodes"]
    tail = slice(T - 9, T + 1)

    def run(d, att, lam):
        return next(r for r in d["runs"] if r["case"] == "dfl" and r["attacker"] == att and r["lam"] == lam)

    print("== clean quality, last 10 rounds, mean ± sd over seeds (pp) ==")
    print(f"  {'defense':8s} | q local-val     | audit balanced  | audit min-node  | q vs uniform | audit vs uniform")
    base = {}
    for name, d in data.items():
        seeds = d["config"]["seeds"]
        clean = [next(r for r in d["runs"] if r["case"] == "dfl" and r["attacker"] is None and r["seed"] == sd)
                 for sd in seeds]
        q = np.array([np.asarray(r["q"])[tail].mean() for r in clean])
        a = np.array([np.asarray(r["audit"])[tail].mean() for r in clean])
        amin = np.array([np.asarray(r["audit"])[tail].mean(0).min() for r in clean])
        if name == "uniform":
            base = {"q": q, "a": a}
        print(f"  {name:8s} | {pm(q)} | {pm(a)} | {pm(amin)} | {100 * (q - base['q']).mean():+12.2f} |"
              f" {100 * (a - base['a']).mean():+16.2f}")
    local = [r for r in data["uniform"]["runs"] if r["case"] == "local"]
    print(f"  {'local':8s} | {pm([np.asarray(r['q'])[tail].mean() for r in local])} |"
          f" {pm([np.asarray(r['audit'])[tail].mean() for r in local])} |"
          f" {pm([np.asarray(r['audit'])[tail].mean(0).min() for r in local])} |")

    lams_show = (2.0, 4.0, 8.0, 16.0)
    for key, label in (("Dss", "neighbour qbar drop, mean over attacked rounds"),
                       ("own", "attacker's own q drop, mean over attacked rounds"),
                       ("net", "honest nodes' balanced-audit drop, last 10 rounds")):
        print(f"\n== damage: {label} (pp; attacker 5 interior / 1 edge / 0 corner) ==")
        print(f"  {'defense':8s} | " + " | ".join(f"lam={l:<4g} 5 / 1 / 0      " for l in lams_show))
        for name, d in data.items():
            seeds = d["config"]["seeds"]
            cells = []
            for lam in lams_show:
                vals = []
                for att in (5, 1, 0):
                    h = list(nb[att])
                    honest = np.arange(n) != att
                    x = []
                    for sd in seeds:
                        c = next(r for r in d["runs"] if r["case"] == "dfl" and r["attacker"] is None and r["seed"] == sd)
                        r = next(r for r in d["runs"] if r["case"] == "dfl" and r["attacker"] == att
                                 and r["lam"] == lam and r["seed"] == sd)
                        dq = np.asarray(c["q"]) - np.asarray(r["q"])
                        da = np.asarray(c["audit"]) - np.asarray(r["audit"])
                        x.append({"Dss": dq[s + 1:][:, h].mean(), "own": dq[s + 1:, att].mean(),
                                  "net": da[tail][:, honest].mean()}[key])
                    vals.append(100 * np.mean(x))
                cells.append(" / ".join(f"{v:5.1f}" for v in vals))
            print(f"  {name:8s} | " + " | ".join(f"{c:21s}" for c in cells))

    K = T - s
    w = args.discount ** np.arange(K)
    w /= w.sum()
    print(f"\n== B3 critical cost c* (metric {args.metric}); '-' = no significantly profitable lambda;"
          " (lambda at c just below c*) ==")
    print(f"  {'defense':8s} | {'att':3s} | {'one-shot':14s} | " + " | ".join(f"rho={r:<4g}       " for r in RHOS))
    summary = {}
    for name, d in data.items():
        cur = curves(d, args.metric)
        lams = np.array(sorted({l for _, l in cur}))
        for att in (5, 1, 0):
            rows = [np.stack([cur[(att, l)]["D"][:, 0] for l in lams], 1)]
            rows += [np.stack([(cur[(att, l)]["D"] - r * cur[(att, l)]["Own"]) @ w for l in lams], 1) for r in RHOS]
            cells = []
            for per_seed in rows:
                v = significant_mean(per_seed)
                ratio = v / lams ** 2
                k = int(np.argmax(ratio))
                cells.append(f"{ratio[k]:8.1e} ({lams[k]:g})" if ratio[k] > 0 else f"{'-':>8}     ")
            summary[(name, att)] = cells
            print(f"  {name:8s} | {att:3d} | {cells[0]:14s} | " + " | ".join(f"{c:14s}" for c in cells[1:]))

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for name, d in data.items():
        lams = sorted({r["lam"] for r in d["runs"] if r["attacker"] == 5})
        seeds = d["config"]["seeds"]
        for k, key in enumerate(("Dss", "own", "net")):
            ys = []
            for lam in lams:
                x = []
                for sd in seeds:
                    c = next(r for r in d["runs"] if r["case"] == "dfl" and r["attacker"] is None and r["seed"] == sd)
                    r = next(r for r in d["runs"] if r["attacker"] == 5 and r["lam"] == lam and r["seed"] == sd)
                    dq = np.asarray(c["q"]) - np.asarray(r["q"])
                    da = np.asarray(c["audit"]) - np.asarray(r["audit"])
                    x.append({"Dss": dq[s + 1:][:, list(nb[5])].mean(), "own": dq[s + 1:, 5].mean(),
                              "net": da[tail][:, np.arange(n) != 5].mean()}[key])
                ys.append(100 * np.mean(x))
            ax[k].plot([0, *lams], [0, *ys], marker="o", label=name)
    for k, t in enumerate(("neighbour qbar drop (Dss)", "attacker's own q drop", "honest audit drop (last 10)")):
        ax[k].set_xscale("symlog", linthresh=1); ax[k].set_xlabel("lambda"); ax[k].set_ylabel("pp lost")
        ax[k].set_title(f"{t}, attacker 5"); ax[k].axhline(0, color="gray", lw=0.5)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "b5_lattice_noniid_damage.png", dpi=120)
    print(f"\nfigure: {RESULTS / 'b5_lattice_noniid_damage.png'}")


if __name__ == "__main__":
    main()
