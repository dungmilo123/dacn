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

## Matched policy ablations (M003)

The M003 ablation panel asks a narrow, falsifiable question: does the
artifact-backed Bayesian policy's measured contribution come from its *belief*
(evidence-conditioned posterior) or from its *strategic* attacker model? It
answers by re-deriving three artifacts through the **same production
calibration path** and evaluating all eight panel entries in one matched
paired study. It is not a claim that any ablation improves robustness.

### 1. Re-derive the three variants (one source protocol, three flags)

`configs/calibrate_ablation_chain3.yaml` is a single frozen source calibration
protocol. The `--variant` flag selects the derivation assumptions; without it
the config's own `variant` is used (default `full`). The override is
revalidated through the strict `CalibrationConfig` model, so an invalid
variant/mixture combination is rejected before anything is collected:

```bash
uv run dfl calibrate configs/calibrate_ablation_chain3.yaml \
    --output runs/ablation/full         --variant full
uv run dfl calibrate configs/calibrate_ablation_chain3.yaml \
    --output runs/ablation/no_belief    --variant no_belief
uv run dfl calibrate configs/calibrate_ablation_chain3.yaml \
    --output runs/ablation/no_strategic --variant no_strategic
```

Each command prints exactly one bounded stdout success marker including the
published `aggregate_digest`; the variant and its assumptions live in the
artifact's digest-covered `derivation` metadata, not in a config dump.

What each variant changes (D036/D037):

* **`full`** is the ordinary Bayesian artifact (pinned source attacker mixture
  `(0, 2, 5)`, evidence-conditioned prior and likelihood).
* **`no_belief`** re-estimates over the *unchanged* full schedule but assembles
  the prior **uniformly over every feasible per-degree joint state** and the
  likelihood **uniformly over the complete evidence support** for both local
  types in every context. The posterior is therefore exactly uniform over
  feasible states, independent of the observed evidence; the payoff tables are
  byte-identical to `full`. The coverage metadata marks posterior calibration
  explicitly non-applicable, so a full-policy Brier score is never presented as
  meaningful no-belief evidence.
* **`no_strategic`** uses a **fixed `kappa=5` (strong) attacker on every
  attacked episode** and re-estimates prior/likelihood/payoff from those new
  records. The source dataset/partition/design membership, episode
  seed/strata/placements and candidate budgets stay equal; the *calibration
  attack composition and trajectories* change. This is a fixed-mixture /
  fixed-behavior isolation, **not** an attacker-response optimizer (none
  exists in this codebase). It removes attack-mixture variability; it does not
  search for a best response.

All three artifacts share one path-independent calibration-config identity and
identical actual dataset/partition/design/schedule-pairing fingerprints, so
they carry exactly one `matched_calibration_input_digest`. Their aggregate
digests are distinct.

> **Fresh-full requirement.** The `full` entry of an ablation study must be a
> *freshly calibrated* artifact carrying `derivation` metadata. A legacy full
> artifact without derivation metadata still loads and runs in an ordinary
> six-policy study, but it is rejected for ablation comparison (missing
> provenance), because there is no way to prove it was derived from the same
> calibration inputs as the ablations.

### 2. Build an ablation-enabled held-out spec

The published digest of each artifact is in its success marker. To extract the
two identities the spec pins (the per-variant aggregate digest and the shared
matched-input digest):

```bash
uv run python - <<'PY'
from dfl_sim import load_bayesian_artifact, matched_calibration_input_digest
for name in ("full", "no_belief", "no_strategic"):
    a = load_bayesian_artifact(f"runs/ablation/{name}")
    print(name, a.manifest.aggregate_digest)
print("matched", matched_calibration_input_digest(
    load_bayesian_artifact("runs/ablation/full").manifest.derivation))
PY
```

An ablation-enabled held-out spec is the ordinary held-out spec plus an
optional `ablations` block and exactly the **eight** panel ids
(`EXTENDED_POLICY_IDS`): the six S01 baselines plus `bayesian_no_belief` and
`bayesian_no_strategic`. The three artifact-backed entries
(`bayesian`, `bayesian_no_belief`, `bayesian_no_strategic`) must point at
distinct artifact directories and each must declare the same matched tuning
budget/selection provenance as every other tunable policy:

```yaml
study:
  study_id: ablation_v1
  data_provenance: injected          # or real_mnist for the production path
  base: {...}
  seeds: [11, 12, 13]                # selection seeds
  candidate_budget: 3
  selection_seed: 7
  selection_provenance: selection-seed-budget
  panel:
    - {id: local_only, ...}
    - {id: plain_average, ...}
    - {id: fixed_similarity, ...}
    - {id: trust_weighted, ...}
    - {id: balance, ...}
    - {id: bayesian, aggregation: {kind: bayesian, artifact_dir: runs/ablation/full, ...}, ...}
    - {id: bayesian_no_belief, aggregation: {kind: bayesian, artifact_dir: runs/ablation/no_belief, ...}, ...}
    - {id: bayesian_no_strategic, aggregation: {kind: bayesian, artifact_dir: runs/ablation/no_strategic, ...}, ...}
  ablations:
    matched_calibration_input_digest: <shared 64-hex>
    expected_aggregate_digests:
      - {variant: full, expected_aggregate_digest: <full 64-hex>}
      - {variant: no_belief, expected_aggregate_digest: <no_belief 64-hex>}
      - {variant: no_strategic, expected_aggregate_digest: <no_strategic 64-hex>}
final_seeds: [21, 22]                # disjoint from study.seeds
known_schedules: [{id: periodic_known, schedule: {kind: periodic}}]
held_out_schedule: {id: held_out_schedule, schedule: {kind: held_out}}
```

`data_provenance` must agree with the artifacts: a study declaring `injected`
requires `injected`-provenance artifacts (and an explicit in-memory dataset
pair at the runner seam), while `real_mnist` requires `production`-provenance
artifacts. Without the `ablations` block the study is the legacy six-policy
contract and its canonical bytes/digests are unchanged.

### 3. Evaluate (selection -> freeze -> final, resume)

```bash
uv run dfl evaluate ABLATION_HELD_OUT.yaml --output runs/ablation_study
```

The command is the same `dfl evaluate` held-out path. Before the first cell it
loads the three artifacts once and runs the matched-artifact preflight, which
verifies pinned aggregate digests, distinct variant identities, source/effective
mixture consistency, the normalized config identity, the actual
dataset/partition/design/schedule-pairing fingerprints, bins, action space,
sampler settings and declared budgets. Only the declared uniform-belief
difference and the fixed-intensity/trajectory difference are whitelisted; the
likelihood/payoff table bytes are never compared across variants. Evaluation
seeds and the runtime attacker schedule are intentionally **not** required to
equal the calibration seed/mixture.

The selection phase runs the known schedules over the selection seeds, freezes
`selection/selection-manifest.json` (recording `selection_spec_digest` and
`status: frozen`), and the final phase re-asserts the disjoint split before
running the final seeds over the known plus reserved unseen schedules. The
matched identity (`matched_calibration_input_digest`, variant, per-variant
aggregate digest, assumption version, and the explicit fixed-behavior
limitation) is propagated into both manifests, the per-cell `cells.jsonl`
checkpoints, and `robustness-summary.json`. A resume revalidates the **current**
pinned artifacts rather than trusting a frozen selection or cached path, so a
replaced or tampered artifact fails closed.

On success stdout carries exactly one bounded marker (`dfl_sim: held out study
... policies=8 ...`); per-cell selection/final progress lines go to stderr with
only the bounded policy id, seed, and schedule id. Any malformed spec, unknown
variant, valid-but-mismatched artifact, missing provenance, or interrupted run
exits nonzero with a concise stderr line and publishes no manifest. Any
published trace is a normal schema-v1 trace and can be replayed:

```bash
uv run dfl view runs/ablation_study/final/traces/held_out_schedule/balance/seed-21
```

### Honest interpretation and evidence class

Equal full/ablated profiles or outcomes are reported as **no measured
contribution on this fixture**; they are never forced into a success narrative
and never stand in for derivation or strategic isolation. The offline
integration test (`tests/test_ablation_cli.py`) is the executable slice demo:
it calibrates all three variants through the real command path and runs a
matched eight-policy held-out study over explicitly **injected** in-process
MNIST-shaped fixtures (no network), then shows a deliberate mismatch rejected
before any cell runs. Actual downloaded-MNIST execution is S04 evidence, and the
canonical report figures/cost framing are S05; neither is claimed here.

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
