# dfl-sim

Deterministic synchronous Decentralized Federated Learning (DFL) simulator.

M001 delivers a correct, reproducible synchronous DFL kernel with a minimal
canonical round log, a privileged post-commit evaluator, and a versioned
trajectory store. It deliberately ships no defense algorithm.

## Quickstart

```bash
uv sync

# S04 production command: publish a complete schema-v1 trace directory.
uv run dfl run configs/clean_3node.yaml --output runs/clean

# S01 baseline study: publish one paired schema-v1 trace per policy/seed cell
# plus a versioned baseline manifest.
uv run dfl evaluate STUDY.yaml --output runs/baseline_study

# S02 held-out adaptive study: selection -> freeze -> disjoint final seeds
# (including the reserved unseen schedule) in one atomic command.
uv run dfl evaluate HELD_OUT.yaml --output runs/held_out_study

# S05 read-only replay viewer for a completed trace (localhost only).
uv run dfl view runs/clean --port 8050

# Backward-compatible minimal JSONL round log.
uv run python -m dfl_sim configs/clean_3node.yaml --log run.jsonl
```

`dfl run` refuses to overwrite an existing output path, validates the config and
destination before training, and reports the published trace directory with its
round and node counts on success. Any config, data, evaluator, trajectory, or
filesystem failure exits nonzero with a concise stderr message and leaves no
partial trace behind.

## Trace layout (schema v1)

A published trace directory contains exactly four files:

```
runs/clean/
├── config.json     # canonical JSON snapshot of the resolved ExperimentConfig
├── events.jsonl    # one canonical evaluated-round event per round
├── tensors.npz     # initial plus cadence-selected committed-state snapshots
└── manifest.json   # node/edge/type metadata, counts, and artifact digests
```

`config.json`, `events.jsonl`, and `tensors.npz` are byte-for-byte reproducible
for the same config and dataset: no timestamps, UUIDs, hostnames, or absolute
paths are recorded. `manifest.json` carries the SHA-256 digest of each artifact
so downstream consumers (S05) can validate integrity before reading.

## Configuration

Configuration is strict YAML validated with Pydantic: unknown keys are
rejected, every numeric value is bounded, and no config value selects or
executes Python code. See `configs/` for examples.

`evaluation.max_samples` / `evaluation.batch_size` bound the privileged
post-commit evaluator (defaults `512` / `64`), and `trace.snapshot_cadence`
controls how many rounds separate committed tensor snapshots (default `1`;
initial and final states are always included). Raising the cadence reduces
trace size at the cost of fewer intermediate snapshots.

## Replay viewer (`dfl view`)

```bash
uv run dfl view runs/clean            # serve on 127.0.0.1:8050
uv run dfl view runs/clean --port 9000
```

`dfl view TRACE_DIR [--port PORT]` is a **read-only, replay-only** viewer for an
already-completed schema-v1 trace. It never trains, never writes to the trace,
and never accepts an upload or a path from the browser.

**Startup and validation semantics.** The trace is fully validated and projected
in memory once, before the server starts:

* The path is opened only through the strict `TraceReader`, so layout, digest,
  canonical-JSON, ZIP-member, and schema-version checks all apply. Tensor
  payloads (`tensors.npz`) are never read, and `config.json` / `events.jsonl` /
  `manifest.json` are never parsed directly.
* A missing, incomplete, unsupported, or tampered trace stops before any socket
  is bound: the CLI prints a concise path-safe message on stderr and exits
  nonzero with no stack trace. A partially projected model is never rendered.
* `--port` is rejected outside `1-65535`. The server always binds to
  `127.0.0.1` only, with the debug console and the reloader disabled.

**Replay controls and visual encodings.**

* **Network** (Cytoscape): deterministic preset (circular) node positions.
  Benign nodes are green ellipses; malicious nodes are red diamonds, both with
  distinct borders and labels. Topology edges are grey undirected lines; the
  current round's directed messages are blue arrow overlays.
* **Round stepping**: `Previous` / `Next` buttons and a discrete slider labeled
  with the actual trace round indices. Stepping is positional and clamps at the
  first/last round, so it can never leave the valid range. A single-round trace
  shows a valid degenerate slider rather than hidden controls.
* **Charts** (Plotly): per-node loss and accuracy histories plotted against the
  true round indices. The node selector emphasizes one node's series (all nodes
  remain visible) and both charts mark the current round with a cursor and the
  selected node's current value.
* **Status text**: a trace/schema/node/round summary, a benign/malicious legend,
  and an accessible current-round plus directed-message status string.

Chart/network redraws are pure in-memory projections of the immutable replay
model, so malformed callback inputs (out-of-range or non-numeric) are clamped
to the deterministic initial round/first node. The 10x breakpoint is browser
payload: thousands of rounds or dense graphs make the full Cytoscape/Plotly
redraw the bottleneck, which is why the projection stores only scalars and
never embeds tensors or duplicate trace content.

## Bayesian defense acceptance (M002)

M002 ships an artifact-backed approximate-myopic Bayesian aggregation policy
and a documented offline calibration path. The held-out chain3 acceptance
sequence is fully offline (MNIST download stays disabled; the automated test
injects deterministic MNIST-shaped in-process data through the existing loader
seams and does not touch the network):

```bash
# 1. Generate the frozen schema-v1 artifact (four tiny reference episodes).
dfl calibrate configs/calibrate_chain3.yaml --output runs/bayesian_chain3_artifact

# 2. Held-out fixed-poison pair: the generated artifact drives the Bayesian
#    policy, and the otherwise-identical local validation-loss reference
#    (tolerance pinned to 0.0) is the comparator.
dfl run configs/bayesian_fixed_poison_3node.yaml --output runs/poison_bayesian --bayesian-report
dfl run configs/validation_reference_fixed_poison_3node.yaml --output runs/poison_reference

# 3. Held-out clean pair: M001's plain-average baseline versus the
#    otherwise-identical Bayesian run.
dfl run configs/bayesian_clean_3node.yaml --output runs/clean_bayesian
dfl run configs/clean_3node.yaml --output runs/clean_plain
```

The fixed-poison pair is byte-for-byte identical in every non-aggregation field
(same seed, topology, data, training, attack, evaluation, and trace settings),
so any selected-profile difference is attributable to aggregation alone. The
clean pair likewise differs only in aggregation. The checked-in acceptance seed
is distinct from every calibration episode seed. The generated artifact must be
produced by `dfl calibrate`; the S01 hand fixture
(`tests/fixtures/bayesian_v1`) is a different artifact and is not accepted by
the generated-artifact location.

Reported numbers:

* `--bayesian-report` prints one canonical `dfl_sim: bayesian_report {...}`
  stdout record only after the trace is atomically published: artifact digest,
  selection count, benign-receiver coverage and support-weighted total-variation
  mismatch, exact fallback reason counts (including explicit zeros) and rate,
  pruning count, candidate totals/max, and elapsed selection-time
  total/min/max. Reporting is observability only: it never writes a file into
  the trace directory and leaves canonical trace bytes unchanged.
* Acceptance gate: final-round mean post-commit evaluation loss for the clean
  Bayesian run must be `<=` the plain-average baseline plus `1e-9`; final mean
  accuracy is reported but is not a second gate. On the fixed-poison pair at
  least one benign `(round, node)` selects different participant-aligned
  multipliers than the validation-loss reference.

The automated end-to-end test is `tests/test_bayesian_acceptance.py`.

## Baseline study (`dfl evaluate`)

```bash
uv run dfl evaluate STUDY.yaml --output runs/baseline_study
```

`dfl evaluate STUDY --output DIR` runs one strict, fairness-enforcing
**baseline study**: every `(seed, policy)` cell is derived from the study's one
frozen base experiment by overriding only `seed` and `aggregation`, so all six
policies of a seed share the same partition, model initialization, and
attacker -- aggregation is the sole experimental variable. It atomically
publishes `baseline-manifest.json` plus one complete schema-v1 trace per cell
under `DIR/traces/<policy>/seed-<n>/`, and refuses to publish anything if even
one cell fails.

The required six-policy panel (stable ids):

| id | Policy |
| --- | --- |
| `local_only` | local-only (no aggregation) |
| `plain_average` | uniform mean of self plus every sender |
| `fixed_similarity` | D017 fixed directional-similarity filter |
| `trust_weighted` | D029 `exp(-beta * max(loss_delta, 0))` trust weighting |
| `balance` | D028 faithful BALANCE (Fang et al., ACM CCS 2024, Algorithm 3) |
| `bayesian` | M002 artifact-backed approximate-myopic Bayesian policy |

**Provenance label.** `data_provenance` is explicit and appears verbatim in
every manifest entry: `real_mnist` consumes the production MNIST loaders, while
`injected` requires the caller to supply an in-memory dataset pair (used by the
offline integration tests, which never touch the network).

**Citation metadata.** Each panel entry carries `citation`/`implementation`
metadata. The `balance` entry is pinned to the D028 external reference
(`venue: ACM CCS 2024`, `identifier: arXiv:2406.10416v4`,
`algorithm: Algorithm 3`); the other entries are project comparators. The
manifest also records each aggregation's `hyperparameters` and a stable
`hyperparameter_digest`, plus the loaded Bayesian `artifact_digest`.

**Matched-budget interpretation.** The study declares one `candidate_budget`,
`selection_seed`, and `selection_provenance`. Every tunable policy
(`fixed_similarity`, `trust_weighted`, `balance`, `bayesian`) must declare the
same positive budget and selection provenance, so no method gets a larger
tuning search than another; the structurally parameter-free baselines
(`local_only`, `plain_average`) must instead declare `mode: exempt` with a zero
budget. A budget or provenance mismatch is rejected before any run.

The manifest is deterministic and environment-independent (artifact paths are
scrubbed to a placeholder and the artifact's own digest is recorded), so two
identical studies produce byte-identical manifest bytes.

On success, stdout carries exactly one stable marker naming the manifest and
output directory (`dfl_sim: baseline study manifest=... output=...`); the
per-cell progress lines go to stderr and contain only the bounded policy id and
seed. Any malformed spec, existing destination, budget mismatch, missing
artifact, or failed cell exits nonzero with a concise stderr line (naming the
policy and seed) and publishes no manifest.

**Inspecting an emitted trace.** Each cell is a normal schema-v1 trace, so any
emitted trace can be replayed with the read-only viewer:

```bash
uv run dfl view runs/baseline_study/traces/balance/seed-11
```

### Held-out adaptive study (same command, held-out spec)

`dfl evaluate` accepts two strict spec kinds from one loader and dispatches on
shape. A baseline spec keeps the S01 behavior exactly; a held-out spec runs the
full **selection -> freeze -> final** study in one atomic command:

```bash
uv run dfl evaluate HELD_OUT.yaml --output runs/held_out_study
# runs/held_out_study/selection/selection-manifest.json
# runs/held_out_study/final/final-manifest.json
```

A held-out spec composes the baseline study as a nested `study` block (its
`seeds` are the *selection* seeds) and adds three fields:

```yaml
study: {study_id: held_out_v1, data_provenance: real_mnist, base: {...},
        seeds: [11, 12], candidate_budget: 3, selection_seed: 7,
        selection_provenance: selection-seed-budget, panel: [...]}
final_seeds: [21]                 # disjoint from study.seeds
known_schedules:                  # adaptive schedules selection may build
  - {id: periodic_known, schedule: {kind: periodic}}
held_out_schedule:                # reserved, unseen; only final builds it
  {id: held_out_schedule, schedule: {kind: held_out}}
```

The nested `study` (including the six-policy panel and matched-budget rules) is
exactly the S01 baseline contract.

**Split and no-leak semantics.** The selection runner is typed to accept only a
`SelectionSpec`, a projection that structurally carries no `final_seeds` or
`held_out_schedule`, so final seeds and the reserved schedule are unreachable
during selection. The frozen `selection/selection-manifest.json` records the
`selection_spec_digest` and `status: frozen`. The final runner re-asserts the
disjoint split, refuses a missing/incomplete/non-frozen selection output, and
fails closed -- naming only the bounded digests -- when the selection-side spec
was mutated after freezing.

**Adaptive schedule provenance.** Every schedule is a pure per-round kappa map
keyed by a domain-separated stable seed, so a cell's realized attack is
identical across the six policies and reproducible across processes. Each
selection/final manifest entry records the `schedule`, its `schedule_digest`,
and the exact `realized_kappa_sequence` each policy realized through the
unchanged `OutgoingController` seam; `held_out: true` marks entries from the
reserved unseen schedule.

**Success marker.** stdout carries exactly one bounded marker naming both
manifests, the output root, the panel size, the seed counts, and the cell
counts (`dfl_sim: held out study selection_manifest=... final_manifest=...
output=... policies=6 selection_seeds=2 final_seeds=1 selection_cells=12
final_cells=12`); per-cell selection/final progress lines go to stderr and
contain only the bounded policy id, seed, and schedule id. Any failure removes
the private staging directory and publishes nothing.

**Final-only replay.** To rerun only the final phase against an existing frozen
selection output:

```bash
uv run dfl evaluate HELD_OUT.yaml --output runs/held_out_final \
  --selection-output runs/held_out_study/selection
```

`--selection-output` is rejected for a baseline spec. `--output` then receives
`final-manifest.json` directly instead of a two-phase study root.

**Inspecting an emitted trace.** Selection and final cells are normal
schema-v1 traces, so any can be replayed:

```bash
uv run dfl view runs/held_out_study/selection/traces/periodic_known/balance/seed-11
uv run dfl view runs/held_out_study/final/traces/held_out_schedule/balance/seed-21
```

## Testing

Run the ordinary offline suite (no browser binary required):

```bash
uv run pytest -q
```

This covers the projection, Dash layout/callbacks, and CLI. The viewer also has
an **opt-in real-browser smoke** that proves Cytoscape and Plotly actually
hydrate and that the round stepping controls work in Chromium. It is skipped
unless `DFL_RUN_BROWSER_TESTS=1` is set, so the default suite and any CI without
a browser stay green.

Browser-smoke preconditions:

```bash
uv sync                                # installs the pytest-playwright dev dep
uv run playwright install chromium     # one-time browser download
# 127.0.0.1 loopback must be available (the server binds an ephemeral port)
```

Run just the browser smoke:

```bash
DFL_RUN_BROWSER_TESTS=1 uv run pytest -q tests/test_viewer_browser.py --browser chromium
```

Run the entire suite including the browser smoke:

```bash
DFL_RUN_BROWSER_TESTS=1 uv run pytest -q
```

When `DFL_RUN_BROWSER_TESTS=1` is set the smoke fails rather than silently
skipping if Chromium is missing or the app fails to hydrate. It navigates only
to its own ephemeral `127.0.0.1` URL, and the test server is always shut down
even when a browser assertion fails.
