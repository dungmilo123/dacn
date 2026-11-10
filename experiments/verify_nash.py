"""Re-test Nash profiles from b4_nash.py on fresh seeds (selection-bias check).

The search picks, per node, the best of 8 alternatives at T_CRIT on 5 seeds, so some switches can
be false positives. For each attacker i of a saved profile, this compares U_i(profile) against
U_i(profile with lambda_i = 0) on holdout seeds never used in the search, and reports the mean
paired gain, its SE and whether it stays significant. It also re-measures quality of all-honest
vs the profile on the holdout seeds.
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import b4_nash as b
from dfl_lambda import lattice


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=[61, 67, 71, 73, 79])
    p.add_argument("--workers", type=int, default=min(22, os.cpu_count() or 1))
    args = p.parse_args()
    for path in args.inputs:
        d = json.loads(path.read_text())
        cfg = d["config"]
        sim = {k: cfg.get(k, dflt) for k, dflt in (
            ("nodes", 16), ("mode", "non_iid"), ("alpha", 1.0), ("rounds", 40), ("attack_start", 10),
            ("samples", 1024), ("validation", 200), ("audit_per_class", 100), ("hidden", 64), ("batch_size", 64),
            ("lr", 0.1), ("defense", "uniform"), ("gate_tol", 1.0), ("trim_b", 1), ("balance_gamma", 0.3),
            ("balance_kappa", 1.0), ("balance_alpha", 0.5), ("flag_tol", 0.5), ("trust_beta", 0.5),
            ("trust_thr", 0.75), ("trust_warmup", 5))}
        nb = lattice(sim["nodes"])
        K = sim["rounds"] - sim["attack_start"]
        w = cfg["discount"] ** np.arange(K)
        w /= w.sum()
        print(f"\n### {path.name}: defense={sim['defense']} metric={cfg['metric']} holdout seeds {args.seeds}")
        with Pool(args.workers, initializer=b._init, initargs=(cfg["cache"], sim)) as pool:
            game = b.Game(pool, args.seeds, cfg["metric"], nb)
            for res in d["results"]:
                prof = tuple(res["profile"])
                rho, c = res["rho"], res["c"]
                atts = [i for i, l in enumerate(prof) if l > 0]
                devs = [tuple(0.0 if j == i else prof[j] for j in range(len(prof))) for i in atts]
                out = game.outcomes([prof, *devs])
                print(f"  rho={rho} c={c:g} profile {[f'{l:g}' for l in prof]}")
                kept = 0
                for i, dev in zip(atts, devs):
                    g = game.utility(out[prof], i, prof[i], rho, c, w) - game.utility(out[dev], i, 0.0, rho, c, w)
                    se = g.std(ddof=1) / np.sqrt(len(g))
                    sig = g.mean() > b.T_CRIT * se
                    kept += sig
                    print(f"    node {i:2d} lambda {prof[i]:g}: gain vs honest {g.mean():+.5f} (SE {se:.5f})"
                          f" {'significant' if sig else 'NOT significant'}")
                q = pool.map(b.simulate, [(sd, pr, True) for pr in ((0.0,) * len(prof), prof) for sd in args.seeds])
                n = len(args.seeds)
                print(f"    attacks still significant on holdout: {kept}/{len(atts)};"
                      f" audit last10 honest {100 * np.mean([r[1][-10:].mean() for r in q[:n]]):.2f}"
                      f" -> profile {100 * np.mean([r[1][-10:].mean() for r in q[n:]]):.2f};"
                      f" q {100 * np.mean([r[0][-10:].mean() for r in q[:n]]):.2f}"
                      f" -> {100 * np.mean([r[0][-10:].mean() for r in q[n:]]):.2f}")


if __name__ == "__main__":
    main()
