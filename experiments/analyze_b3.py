"""B3: attacker best response lambda* under u_i = a1 q_i + a2 qbar_i + a3 (q_i - qbar_i) - g k lambda^2.

No new training: uses the paired B2 runs (clean DFL vs one attacker with constant lambda).
Gain of attacking at lambda relative to honest, divided by (a3 - a2) > 0:

  one-shot : G1(lambda)     = D1(lambda) - c lambda^2                (own q unaffected in the first round)
  repeated : GR(lambda; rho) = sum_k d^k [D_k - rho Own_k] / sum_k d^k - c lambda^2

  c   = g k / (a3 - a2)              cost per lambda^2 in units of quality
  rho = (a1 + a3) / (a3 - a2) > 1    weight of own-quality loss relative to neighbour harm
  D_k / Own_k = neighbour-mean / own quality drop k rounds after attack start (paired, clean - attacked)

Repeated = stationary lambda from the attack start to the horizon (restricted policy class).
If a3 <= a2 no lambda > 0 is ever profitable and nothing here applies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHOS = (1.0, 1.25, 1.5, 2.0, 3.0)


def curves(d, metric):
    """Per (attacker, lambda): arrays [seed, k] of neighbour damage and own damage, k = rounds after start."""
    cfg, nb = d["config"], d["neighbors"]
    s, n = cfg["attack_start"], cfg["nodes"]
    runs = {(r["seed"], r["case"], r["attacker"], r["lam"]): np.asarray(r[metric]) for r in d["runs"]}
    seeds = cfg["seeds"]
    out = {}
    for (sd, case, att, lam), q in runs.items():
        if att is None:
            continue
        clean = runs[(sd, "dfl", None, 0.0)]
        diff = clean - q
        nbar = diff[:, list(nb[att])].mean(1)
        if not np.all(diff[: s + 2, att] == 0):  # own model is untouched until the second attacked round
            raise AssertionError(f"own quality changed too early: seed {sd}, attacker {att}, lambda {lam}")
        out.setdefault((att, lam), {"D": [], "Own": []})
        out[(att, lam)]["D"].append(nbar[s + 1:])
        out[(att, lam)]["Own"].append(diff[s + 1:, att])
    seeds_order = len(seeds)
    for v in out.values():
        v["D"], v["Own"] = np.stack(v["D"]), np.stack(v["Own"])
        assert len(v["D"]) == seeds_order
    return out


T_CRIT = 2.776  # one-sided 97.5% Student t, 4 df (5 paired seeds)


def significant_mean(per_seed):
    """Mean gain per lambda; lambdas whose gain is not > T_CRIT * SE across seeds become -inf
    (excluded), so seed noise at small lambda cannot fake a profitable attack."""
    mean = per_seed.mean(0)
    se = per_seed.std(0, ddof=1) / np.sqrt(len(per_seed))
    return np.where(mean > T_CRIT * se, mean, -np.inf)


def best(lams, values, cost_c):
    """lambda* and gain over the grid {0} U lams; ties -> smaller lambda (honest preferred)."""
    lams = np.concatenate([[0.0], lams])
    gains = np.concatenate([[0.0], values]) - cost_c * lams ** 2
    k = int(np.argmax(gains))  # first max = smallest lambda
    return lams[k], gains[k]


def summarize(d, metric, discount, label, fig_prefix):
    cur = curves(d, metric)
    attackers = sorted({a for a, _ in cur})
    lams = np.array(sorted({l for _, l in cur}))
    K = next(iter(cur.values()))["D"].shape[1]
    w = discount ** np.arange(K)
    w /= w.sum()
    cs = np.logspace(-6, -1, 101)
    print(f"\n######## q = {label}; discount {discount} over {K} attacked rounds ########")

    def one_shot(att):
        """[seed, lambda] first-round neighbour damage."""
        return np.stack([cur[(att, l)]["D"][:, 0] for l in lams], 1)

    def repeated(att, rho):
        """[seed, lambda] discounted mean of neighbour damage minus rho * own damage."""
        return np.stack([(cur[(att, l)]["D"] - rho * cur[(att, l)]["Own"]) @ w for l in lams], 1)

    nseeds = next(iter(cur.values()))["D"].shape[0]
    print(f"V(lambda) = mean gain over {nseeds} seeds, kept only if > {T_CRIT} SE (else lambda excluded)")
    print("critical cost c* = max_lambda V(lambda)/lambda^2: attack profitable iff c < c*;"
          " jump = lambda* at c slightly below c*; '-' = no lambda significantly profitable")
    header = "  attacker | one-shot c*  jump | " + " | ".join(f"rho={r:<4} c*  jump" for r in RHOS)
    print(header)
    table = {}
    for att in attackers:
        row = []
        for name, per_seed in [("one", one_shot(att))] + [(r, repeated(att, r)) for r in RHOS]:
            v = significant_mean(per_seed)
            ratio = v / lams ** 2
            k = int(np.argmax(ratio))
            cstar = max(ratio[k], 0.0)
            table[(att, name)] = (cstar, lams[k] if cstar > 0 else 0.0)
            row.append(f"{cstar:9.2e} {lams[k]:4g}" if cstar > 0 else f"{'-':>9} {'':4}")
        print(f"  {att:8d} | " + " | ".join(row))
    att = 5 if 5 in attackers else attackers[0]
    print(f"  V(lambda) x100 for attacker {att} (mean over seeds; * = significant):")
    print("    lambda | one-shot | " + " | ".join(f"rho={r:<4}" for r in RHOS))
    rows = [one_shot(att)] + [repeated(att, r) for r in RHOS]
    for j, l in enumerate(lams):
        cells = []
        for per_seed in rows:
            m, ok = per_seed[:, j].mean(), np.isfinite(significant_mean(per_seed)[j])
            cells.append(f"{100 * m:7.2f}{'*' if ok else ' '}")
        print(f"    {l:6g} | " + " | ".join(cells))

    # lambda*(c) step for the one-shot game, every attacker position.
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for att in attackers:
        v = significant_mean(one_shot(att))
        ax[0].step(cs, [best(lams, v, c)[0] for c in cs], where="post", label=f"attacker {att} (deg {len(d['neighbors'][att])})")
    ax[0].set_xscale("log"); ax[0].set_xlabel("c = gamma kappa / (a3 - a2)"); ax[0].set_ylabel("lambda*")
    ax[0].set_title(f"one-shot best response ({label})"); ax[0].legend(fontsize=8)
    # (c, rho) phase map for the interior node and the corner node.
    rhos = np.linspace(1, 3, 81)
    for k, att in enumerate((5, 0)):
        if att not in attackers:
            continue
        grid = np.array([[best(lams, significant_mean(repeated(att, r)), c)[0] for c in cs] for r in rhos])
        im = ax[k + 1].pcolormesh(cs, rhos, grid, shading="auto", cmap="viridis")
        ax[k + 1].set_xscale("log"); ax[k + 1].set_xlabel("c"); ax[k + 1].set_ylabel("rho = (a1+a3)/(a3-a2)")
        ax[k + 1].set_title(f"repeated lambda*, attacker {att} ({label})")
        fig.colorbar(im, ax=ax[k + 1], label="lambda*")
    fig.tight_layout()
    path = f"{fig_prefix}_{metric}.png"
    fig.savefig(path, dpi=120)
    print(f"  figure: {path}")

    # The jump: interior node, one-shot, around c*.
    att = 5 if 5 in attackers else attackers[0]
    cstar = table[(att, "one")][0]
    if cstar > 0:
        v = significant_mean(one_shot(att))
        print(f"  one-shot jump, attacker {att}: " + ", ".join(
            f"c={f * cstar:.2e} -> lambda*={best(lams, v, f * cstar)[0]:g}" for f in (2, 1.01, 0.99, 0.5, 0.1)))
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("experiments/results/b1_b2_lattice_noniid.json"))
    p.add_argument("--discount", type=float, default=0.95)
    args = p.parse_args()
    d = json.loads(args.input.read_text())
    prefix = str(args.input.with_suffix("")).replace("b1_b2", "b3")
    print(f"lambdas {sorted({r['lam'] for r in d['runs'] if r['attacker'] is not None})}, seeds {d['config']['seeds']}")
    summarize(d, "q", args.discount, "local validation (main)", prefix)
    summarize(d, "audit", args.discount, "balanced audit (sensitivity)", prefix)


if __name__ == "__main__":
    main()
