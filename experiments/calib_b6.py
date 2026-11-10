"""B6 calibration: val-loss excess of honest vs manipulated messages (receiver's own validation).

Input = recip run with trust_thr=0 (never cuts; aggregation identical to gate), which logs
excess_ij = val-loss(m_j) - val-loss(own local) for every directed delivery and round.
Honest = messages from clean runs (all senders) at rounds >= attack_start.
Attacked = messages sent by the attacker in attacked runs at rounds >= attack_start.
"""
import json
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/results/b6_calib_excess_lattice_noniid.json")
d = json.loads(path.read_text())
s = d["config"]["attack_start"]
honest, attacked = [], {}
for r in d["runs"]:
    if r["case"] != "dfl":
        continue
    for t, log in enumerate(r["links"]):
        if t < s:
            continue
        for i, j, e in log:
            if r["attacker"] is None:
                honest.append(e)
            elif j == r["attacker"]:
                attacked.setdefault(r["lam"], []).append(e)
honest = np.array(honest)
print(f"honest messages (clean runs, t>={s}): n={len(honest)}")
for p in (50, 90, 95, 99, 99.5, 99.9, 100):
    print(f"  p{p:<5} excess = {np.percentile(honest, p):8.4f}")
tols = (0.1, 0.2, 0.3, 0.5, 1.0)
print("\nflag rate (excess > tol): honest | attacker message per lambda")
print("  tol  | honest  | " + " | ".join(f"lam={l:<4g}" for l in sorted(attacked)))
for tol in tols:
    cells = [f"{100 * np.mean(np.asarray(attacked[l]) > tol):7.1f}%" for l in sorted(attacked)]
    print(f"  {tol:4.2f} | {100 * np.mean(honest > tol):6.2f}% | " + " | ".join(cells))
