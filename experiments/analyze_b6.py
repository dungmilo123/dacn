"""B6 link diagnostics for the recip defense (rounds >= attack_start).

inbound-to-attacker : share of (neighbour -> attacker) deliveries that happened; low = attacker cut off
honest links (clean): share of honest directed deliveries that happened with no attacker; 1 - this = false cutoffs
attacker msgs kept  : share of the attacker's delivered messages that passed the aggregation gate
"""
import json
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/results/b6_recip_lattice_noniid.json")
d = json.loads(path.read_text())
cfg = d["config"]
s, tol = cfg["attack_start"], cfg["gate_tol"]
runs = [r for r in d["runs"] if r["case"] == "dfl"]

clean = [r for r in runs if r["attacker"] is None]
deliv = [e is not None for r in clean for log in r["links"][s:] for _, _, e in log]
print(f"clean runs: honest directed deliveries {100 * np.mean(deliv):.2f}% "
      f"({len(deliv) - sum(deliv)} cut of {len(deliv)})")

print("\n attacker | lambda | inbound-to-attacker delivered | attacker msgs delivered | delivered attacker msgs kept by gate")
for att in sorted({r["attacker"] for r in runs if r["attacker"] is not None}):
    for lam in sorted({r["lam"] for r in runs if r["attacker"] == att}):
        inb, out, kept = [], [], []
        for r in (r for r in runs if r["attacker"] == att and r["lam"] == lam):
            for log in r["links"][s:]:
                for i, j, e in log:
                    if i == att:
                        inb.append(e is not None)
                    if j == att:
                        out.append(e is not None)
                        if e is not None:
                            kept.append(e <= tol)
        print(f" {att:8d} | {lam:6g} | {100 * np.mean(inb):27.1f}% | {100 * np.mean(out):22.1f}% |"
              f" {100 * np.mean(kept) if kept else float('nan'):35.1f}%")
