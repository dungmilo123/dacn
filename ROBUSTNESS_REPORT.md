# Canonical robustness report (M003/S05)

Measured findings from the frozen nine-condition canonical matrix executed on
**actually downloaded MNIST** and independently audited at M003/S05/T07. This
document describes what was observed; it does not claim an improvement,
statistical confidence, or a strategic equilibrium, and it does not quote any
injected CI fixture as a real result.

## What was run

| Dimension | Value |
|---|---|
| Command | `.venv/bin/python -m dfl_sim.robustness_study configs/robustness_mnist.yaml --output runs/m003-canonical-robustness` |
| Data | real downloaded MNIST (deep-verified cache; `deep_cache_verified=true`) |
| Partition | capacity-constrained class-wise Dirichlet, `alpha in {0.1, 0.5, 1.0}` |
| Topology | Watts-Strogatz, `n=10`, `k=2`, `p=0.3`, 6 rounds, seed-derived |
| Attacker placement | `low`, `mid`, `high` realized degree rank (seed-derived) |
| Policies | 8: `local_only`, `plain_average`, `fixed_similarity`, `trust_weighted`, `balance`, `bayesian`, `bayesian_no_belief`, `bayesian_no_strategic` |
| Schedules | `duty_cycle`, `periodic`, `delayed_onset`, `random` + final-only `held_out` |
| Selection seeds | 1101, 1102 |
| Final seeds | 2101, 2102, 2103 (disjoint from selection) |
| Calibration seed | 20240101 |
| Training / evaluation | 64 samples/node / 128 samples, CPU, `threads: 1` |
| Cells | 576 selection, 1080 final, 360 aggregates, 630 paired deltas, 1080 replay links |

Verified identities: `protocol_digest=9aafbe5c…`, `report_digest=43a96aea…`,
`audit_digest=2fbe59d8…`, cache manifest `a330b74d…`.

Check the artifacts yourself:

```bash
.venv/bin/python -m dfl_sim report runs/m003-canonical-robustness --verify-only
.venv/bin/python -m dfl_sim.canonical_audit runs/m003-canonical-robustness
```

Report and figures: `runs/m003-canonical-robustness/report/canonical-report.json`
and `runs/m003-canonical-robustness/report/figures/` (five deterministic SVGs).
Raw acceptance: `runs/m003-canonical-robustness/runtime-audit.json`.

## Measured outcomes

### Solution quality (final accuracy, all eight policies, 45 aggregates each)

| Policy | Mean accuracy | Min aggregate | Max aggregate | Mean worst-node accuracy |
|---|---|---|---|---|
| plain_average | 0.1320 | 0.1198 | 0.1435 | 0.0701 |
| trust_weighted | 0.1301 | 0.1190 | 0.1435 | 0.0698 |
| bayesian_no_belief | 0.1206 | 0.1055 | 0.1331 | 0.0651 |
| bayesian_no_strategic | 0.1203 | 0.1039 | 0.1323 | 0.0651 |
| bayesian | 0.1202 | 0.1060 | 0.1318 | 0.0625 |
| balance | 0.1189 | 0.1029 | 0.1323 | 0.0651 |
| local_only | 0.1189 | 0.1029 | 0.1323 | 0.0651 |
| fixed_similarity | 0.1178 | 0.1018 | 0.1271 | 0.0651 |

Accuracy is weak in absolute terms across every policy and condition. This is a
bounded subset (64 training samples/node, 6 rounds) and is reported as measured,
not tuned.

Per-condition mean accuracy across the eight policies shows the most
heterogeneous partition (alpha 0.1) at the lowest accuracy, with placement
having a smaller effect:

| Condition | Mean accuracy |
|---|---|
| a0p1__low / mid / high | 0.1123 / 0.1099 / 0.1107 |
| a0p5__low / mid / high | 0.1288 / 0.1308 / 0.1292 |
| a1__low / mid / high | 0.1272 / 0.1272 / 0.1249 |

### Paired ablation deltas (mean over 45 condition x schedule pairings)

| Comparison | Mean delta | Min | Max |
|---|---|---|---|
| `bayesian` vs `plain_average` | **-0.01182** | -0.03698 | +0.01120 |
| `bayesian_no_belief` vs `bayesian` | **+0.00047** | -0.00182 | +0.00260 |
| `bayesian_no_strategic` vs `bayesian` | **+0.00019** | -0.00469 | +0.00339 |
| `balance` vs `bayesian` | -0.00124 | -0.00469 | +0.00208 |
| `local_only` vs `bayesian` | -0.00124 | -0.00469 | +0.00208 |
| `fixed_similarity` vs `bayesian` | -0.00238 | -0.00573 | +0.00599 |
| `trust_weighted` vs `bayesian` | +0.00995 | -0.01198 | +0.02656 |

The belief and strategic components are **effectively flat** here: removing
either changes the mean accuracy by less than 0.0005. The defense policies trail
plain averaging on this subset by roughly one to one-and-a-half accuracy points.
These are population means over the declared seeds and conditions — no winner
claim and no significance test is made.

### Seed dispersion and worst nodes

Across the three final seeds the mean **population standard deviation** of
aggregate accuracy is 0.008-0.009 for most policies and 0.016 for
`plain_average` / `trust_weighted`; the mean IQR tracks the same ordering
(0.009-0.019). This is **seed dispersion over three seeds**, explicitly not a
confidence interval.

Worst-node accuracy is materially lower than the mean everywhere. The lowest
single worst-node value observed is 0.0469 (`plain_average`, high- and
mid-placement conditions) versus 0.0547 for `bayesian` and 0.0625 for most
other policies. Reporting the mean alone would hide this.

### Rejection and benign exclusion (weight-based, separate from accuracy)

| Quantity | Count | Rate |
|---|---|---|
| Benign sender opportunities | 102240 | — |
| Benign excluded | 74138 | 0.7251 |
| Delivered opportunities | 115920 | — |
| Malicious active rejected | 9993 | 0.7305 |
| Malicious active accepted | 3687 | 0.2695 |
| Malicious inactive opportunities | 0 | — |

Per-policy malicious-active rejection ranges show the interpretation caveat:
`local_only` and `balance` are 1.0000 (self-only / full-reject behavior),
`fixed_similarity` 0.9467-1.0000, `bayesian` 0.8333-1.0000,
`bayesian_no_belief` 0.8900-1.0000, `bayesian_no_strategic` 0.9500-1.0000, while
`plain_average` and `trust_weighted` are 0.0000 because they always assign
positive weight. This is weight-based exclusion, not a detector
false-positive/true-positive measurement. The high benign exclusion rate
(0.7251) shows the defenses exclude most benign senders under heterogeneity —
the core failure mode R019 asks about.

### Propagation

| Quantity | Value |
|---|---|
| Reached node-rounds | 17605 |
| Total node-rounds | 64800 |
| Reached fraction | 0.2717 |
| Max hop | 6 |
| Min hop | 0 |
| First-reached rounds | 0,1,2,3,4,5 |
| Nodes ever reached (max cell) | 10 |

Poison provenance reaches all 10 nodes in at least one condition/policy cell and
propagates up to the full graph diameter (6 hops) within 6 rounds when the
compromised node is placed at high degree.

### Separate computation and communication costs

Computation is `perf_counter_ns` at the shared round/evaluation seam, CPU,
`threads: 1`, pooled over the run:

| Phase | Nanoseconds | Seconds |
|---|---|---|
| `whole_cell_ns` (outer) | 2683843399422 | 2683.8 |
| `train_ns` | 982925444817 | 982.9 |
| `observation_ns` | 940093805424 | 940.1 |
| `evaluation_ns` | 553906988531 | 553.9 |
| `setup_ns` | 40546866635 | 40.5 |
| `aggregation_ns` | 35490767752 | 35.5 |
| `policy_ns` | 2319456494 | 2.32 |
| `selection_ns` | 652722945 | 0.65 |

Communication is a **logical** count, not physical traffic: actual directed
messages plus the sum of `numel * element_size` over every message's state
tensors at construction, before receiver filtering.

| Quantity | Value |
|---|---|
| Directed messages | 129600 |
| Logical payload bytes | 13840243200 (13.84 GB) |
| Model-message payload bytes (reported separately) | 115335360 (115.3 MB) |

`model_payload_bytes` is a separately aggregated sidecar quantity, not a
substitute for the directed logical payload, and headers/retransmits/network
effects are excluded from both.

Calibration stage wall time (`available=false`), process RSS
(`available=false`), and the historical headline cost channel
(`costs.headline.available=false`) are reported as unavailable because the raw
evidence does not record them. Missing telemetry is never rendered as zero.

## Figures

| Figure | File | What it shows |
|---|---|---|
| accuracy_by_condition | `report/figures/accuracy-by-condition.svg` | Final accuracy by condition and schedule |
| compute_vs_communication | `report/figures/compute-vs-communication.svg` | Computation and communication kept separate |
| paired_ablation_deltas | `report/figures/paired-ablation-deltas.svg` | Paired deltas versus reference policies |
| placement_propagation_worst_node | `report/figures/placement-propagation-worst-node.svg` | Placement, propagation and worst-node outcomes |
| rejection_acceptance | `report/figures/rejection-acceptance.svg` | Non-IID benign exclusions versus malicious acceptance |

## Replay

Selected report-cited traces (all relative links confined to the matrix root):

```bash
# alpha 0.1 (most heterogeneous), low-degree placement, final-only schedule.
.venv/bin/dfl view runs/m003-canonical-robustness/conditions/a0p1__low/study/final/traces/held_out/bayesian/seed-2101

# alpha 1.0 (most IID-like) / high-degree placement.
.venv/bin/dfl view runs/m003-canonical-robustness/conditions/a1__high/study/final/traces/held_out/bayesian/seed-2101
```

Each trace directory is an ordinary four-file schema-v1 trace (config, events,
tensors, manifest) and is cited verbatim by `report.canonical-report.json`.

## Honest limitations

* **Finite-action solution quality only.** No equilibrium, regret bound or
  attacker-response solver is claimed. The Bayesian audit re-enumerates
  candidates under the artifact degree cap; above the cap the variable neighbor
  set is pruned and those rows are counted, not hidden.
* **Three seeds is a population spread, not statistical confidence.** The
  dispersion and IQR above are descriptive.
* **`no_strategic` fixes `kappa`.** It is a fixed-attacker-mixture isolation
  assumption, not a measured strategic response; `no_belief` likewise isolates
  one belief channel under the same action space.
* **Bounded subset.** 10 nodes, 6 rounds, 64 training samples/node, 128
  evaluation samples, one seed-derived Watts-Strogatz graph family. It is not a
  wide parameter sweep and not a physical deployment.
* **Logical communication.** Bytes are uncompressed tensor payload before
  receiver filtering; they are not measured network traffic.
* **Degree ranks may tie.** Low/mid-degree seed-derived graphs contain ties; the
  deterministic node-id tie-break is part of the protocol, and even-`n` middle
  rank is pinned to the lower-middle `(n-1)//2`.
* **No injected evidence is real evidence.** CI fixtures prove mechanism only.
  Every number in this document comes from the downloaded-MNIST matrix above.
* **The S04 IID headline is scoped separately** and is never pooled with these
  non-IID/Watts-Strogatz conditions.

## M003 requirement coverage (evidence audit, T08)

Terminal status updates are owned by slice/milestone closeout; this is an
evidence audit only.

| Req | Evidence in this canonical run |
|---|---|
| R017 shared paired seam | All eight policies run through the same `dfl evaluate`/`LocalObservation` path on identical per-condition partitions, seeds and schedules (`policy_ids`, paired deltas). |
| R018 adaptive schedules | Four named on/off schedules plus the final-only `held_out` schedule, all deterministic and defender-independent (`schedule_ids`). |
| R019 Dirichlet non-IID | `alphas` 0.1/0.5/1.0 with paired seeds; benign exclusion vs malicious acceptance reported separately (`rejection`). |
| R020 topology/placement | 10-node Watts-Strogatz with seed-derived low/mid/high degree placement; propagation and worst-node blocks report the effect. |
| R021 ablations | Per-alpha matched `full`/`no_belief`/`no_strategic` calibration triples evaluated beside the full policy under paired conditions. |
| R022 mean and spread | 360 aggregates over 3 final seeds per condition/schedule, with population std, IQR and worst-node accuracy. |
| R023 computation and communication | Separate `computation` (per-phase `perf_counter_ns`, selection elapsed, fallback counts) and logical `communication` (directed messages and payload bytes) blocks. |
