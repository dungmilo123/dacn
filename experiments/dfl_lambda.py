"""DFL simulator for the utility u_i = a1 q_i + a2 qbar_i + a3 (q_i - qbar_i) - g k lambda^2.

B1: clean DFL vs local-only.  B2: one attacker with intensity lambda; damage curve D(lambda).
B5: same with a receive-side defense (--defense, see aggregate()), applied by every node.

Protocol per round t (synchronous):
  q_i^t   = accuracy of served model w_i^t on node i's local validation set
  l_i     = one local epoch of SGD from w_i^t
  m_i     = l_i - lambda_i^t (l_i - w_i^t)      (lambda=0 honest, 1 = resend w, >1 reversed/amplified)
  w_i^t+1 = aggregate(defense, l_i, {m_j : j in N_i})   (uniform = plain mean; local-only: w_i^t+1 = l_i)
The attacker keeps its own clean l_i; only the message is manipulated.
Training RNG is keyed by (seed, node, round) so runs that differ only in lambda are paired.
Only the 60k official-train file is read; the official test set is never loaded.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path
import struct
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
}


def load_mnist(cache: Path):
    raw = []
    for name, md5 in FILES.items():
        blob = (cache / name).read_bytes()
        if hashlib.md5(blob).hexdigest() != md5:
            raise ValueError(f"MNIST checksum mismatch: {name}")
        raw.append(gzip.decompress(blob))
    magic, count, rows, cols = struct.unpack(">IIII", raw[0][:16])
    if (magic, count, rows, cols) != (2051, 60000, 28, 28):
        raise ValueError("unexpected MNIST image header")
    x = np.frombuffer(raw[0], dtype=np.uint8, offset=16).reshape(count, -1).astype(np.float32) / 255
    y = np.frombuffer(raw[1], dtype=np.uint8, offset=8).astype(np.int64)
    return x, y


def lattice(nodes: int):
    side = int(round(nodes ** 0.5))
    if side * side != nodes:
        raise ValueError("lattice needs a square node count")
    return [tuple(sorted((r + dr) * side + c + dc
                         for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                         if 0 <= r + dr < side and 0 <= c + dc < side))
            for r in range(side) for c in range(side)]


def hops(neighbors, source):
    dist = {source: 0}
    frontier = [source]
    while frontier:
        nxt = []
        for u in frontier:
            for v in neighbors[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return [dist[i] for i in range(len(neighbors))]


def make_split(y, seed, args):
    """Disjoint train / local-validation / balanced audit; Dirichlet non-IID shared by train and val."""
    rng = np.random.default_rng(seed)
    train_pool, val_pool, audit = [], [], []
    for label in range(10):
        rows = rng.permutation(np.flatnonzero(y == label))
        cut = int(0.8 * len(rows))
        train_pool.append(rows[:cut])
        audit.append(rows[cut:cut + args.audit_per_class])
        val_pool.append(rows[cut + args.audit_per_class:])
    clients_train = [[] for _ in range(args.nodes)]
    clients_val = [[] for _ in range(args.nodes)]
    for label in range(10):
        if args.mode == "iid":
            proportions = np.full(args.nodes, 1 / args.nodes)
        else:
            proportions = rng.dirichlet(np.full(args.nodes, args.alpha))
        for pool, clients in ((train_pool[label], clients_train), (val_pool[label], clients_val)):
            counts = rng.multinomial(len(pool), proportions)
            for client, chunk in zip(clients, np.split(pool, np.cumsum(counts)[:-1])):
                client.extend(chunk.tolist())
    train, val = [], []
    for i in range(args.nodes):
        if len(clients_train[i]) < args.samples or len(clients_val[i]) < args.validation:
            raise ValueError(f"node {i} too small: train={len(clients_train[i])}, val={len(clients_val[i])}")
        train.append(rng.choice(clients_train[i], args.samples, replace=False))
        val.append(rng.choice(clients_val[i], args.validation, replace=False))
    audit = np.concatenate(audit)
    every = np.concatenate([*train, *val, audit])
    if len(np.unique(every)) != len(every):
        raise AssertionError("leakage between splits")
    return train, val, audit


def initialize(seed, hidden):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((784, hidden)) * np.sqrt(2 / 784)).astype(np.float32),
            np.zeros(hidden, np.float32),
            (rng.standard_normal((hidden, 10)) * np.sqrt(2 / hidden)).astype(np.float32),
            np.zeros(10, np.float32)]


def predict(model, x):
    with np.errstate(all="ignore"):
        return (np.maximum(x @ model[0] + model[1], 0) @ model[2] + model[3]).argmax(axis=1)


def accuracy(model, x, y):
    return float((predict(model, x) == y).mean())


def train(model, x, y, rng, lr, batch):
    w = [a.copy() for a in model]
    order = rng.permutation(len(y))
    with np.errstate(all="ignore"):
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            xb, yb = x[idx], y[idx]
            pre = xb @ w[0] + w[1]
            h = np.maximum(pre, 0)
            s = h @ w[2] + w[3]
            s -= s.max(axis=1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=1, keepdims=True)
            p[np.arange(len(idx)), yb] -= 1
            p /= len(idx)
            gh = (p @ w[2].T) * (pre > 0)
            for param, grad in zip(w, (xb.T @ gh, gh.sum(0), h.T @ p, p.sum(0))):
                param -= lr * grad
    return w


DEFENSES = ("uniform", "gate", "median", "trimmed", "clip", "balance")


def nll(model, x, y):
    with np.errstate(all="ignore"):
        s = np.maximum(x @ model[0] + model[1], 0) @ model[2] + model[3]
        s = s - s.max(axis=1, keepdims=True)
        v = float((np.log(np.exp(s).sum(axis=1)) - s[np.arange(len(y)), y]).mean())
    return v if np.isfinite(v) else np.inf


def flat(model):
    return np.concatenate([p.ravel() for p in model])


def unflat(vector, like):
    out, k = [], 0
    for p in like:
        out.append(vector[k:k + p.size].reshape(p.shape).astype(np.float32))
        k += p.size
    return out


def aggregate(defense, own, received, xv, yv, t, rounds, a):
    """Receive-side rule of one node: own = its clean local model, received = neighbour messages.

    uniform : mean of {own} U received
    gate    : keep m_j iff val-loss(m_j) <= val-loss(own) + gate_tol; uniform over kept + own
    median  : coordinate-wise median of {own} U received
    trimmed : coordinate-wise mean after dropping the trim_b largest and smallest values
              (b=1; on a degree-2 node 3 values -> identical to median)
    clip    : self-centred clipping, own + mean_{own,j} clip(m_j - own, tau), tau = median_j ||m_j - own||
    balance : BALANCE (Fang et al. 2024): keep m_j iff ||m_j - own|| <= g exp(-k t/T) ||own||;
              w = a own + (1-a) mean(kept); nothing kept -> own
    """
    if defense == "uniform":
        return [np.mean([own[k], *(m[k] for m in received)], axis=0) for k in range(4)]
    if defense == "gate":
        limit = nll(own, xv, yv) + a.gate_tol
        kept = [m for m in received if nll(m, xv, yv) <= limit]
        return [np.mean([own[k], *(m[k] for m in kept)], axis=0) for k in range(4)]
    o = flat(own)
    R = np.stack([flat(m) for m in received])
    with np.errstate(all="ignore"):
        if defense == "median":
            return unflat(np.median(np.vstack([o, R]), axis=0), own)
        if defense == "trimmed":
            S = np.sort(np.vstack([o, R]), axis=0)
            b = a.trim_b if len(S) > 2 * a.trim_b else 0
            return unflat(S[b:len(S) - b].mean(axis=0), own)
        dist = np.linalg.norm(R - o, axis=1)
        if defense == "clip":
            tau = float(np.median(dist))
            scale = np.minimum(1.0, tau / np.maximum(dist, 1e-12))
            return unflat(o + ((R - o) * scale[:, None]).sum(0) / (len(R) + 1), own)
        if defense == "balance":
            keep = dist <= a.balance_gamma * np.exp(-a.balance_kappa * t / rounds) * np.linalg.norm(o)
            if not keep.any():
                return [p.copy() for p in own]
            return unflat(a.balance_alpha * o + (1 - a.balance_alpha) * R[keep].mean(0), own)
    raise ValueError(defense)


# Worker globals (set once per process).
_X = _Y = None


def _init_worker(cache):
    global _X, _Y
    _X, _Y = load_mnist(Path(cache))


def run(job):
    """job: dict(seed, case in {local, dfl}, attacker (int|None), lam, args)."""
    a = argparse.Namespace(**job["args"])
    x, y = _X, _Y
    neighbors = lattice(a.nodes)
    train_idx, val_idx, audit_idx = make_split(y, job["seed"], a)
    w = [initialize(job["seed"], a.hidden) for _ in range(a.nodes)]
    q = np.zeros((a.rounds + 1, a.nodes))
    audit = np.zeros((a.rounds + 1, a.nodes))
    msg_norm = []

    def evaluate(t):
        for i in range(a.nodes):
            q[t, i] = accuracy(w[i], x[val_idx[i]], y[val_idx[i]])
            audit[t, i] = accuracy(w[i], x[audit_idx], y[audit_idx])

    evaluate(0)
    for t in range(a.rounds):
        local = [train(w[i], x[train_idx[i]], y[train_idx[i]],
                       np.random.default_rng([job["seed"], i, t]), a.lr, a.batch_size)
                 for i in range(a.nodes)]
        messages = list(local)
        att = job["attacker"]
        if att is not None and t >= a.attack_start and job["lam"] != 0:
            lam = job["lam"]
            messages[att] = [l - lam * (l - p) for l, p in zip(local[att], w[att])]
            delta = np.sqrt(sum(float(((l - p) ** 2).sum()) for l, p in zip(local[att], w[att])))
            msg_norm.append(abs(lam) * delta)
        if job["case"] == "local":
            w = local
        else:
            w = [aggregate(a.defense, local[i], [messages[j] for j in neighbors[i]],
                           x[val_idx[i]], y[val_idx[i]], t, a.rounds, a) for i in range(a.nodes)]
        evaluate(t + 1)
    return {"seed": job["seed"], "case": job["case"], "attacker": job["attacker"], "lam": job["lam"],
            "q": q.round(6).tolist(), "audit": audit.round(6).tolist(),
            "manip_norm_mean": float(np.mean(msg_norm)) if msg_norm else 0.0,
            "train_label_counts": [np.bincount(y[i], minlength=10).tolist() for i in train_idx],
            "val_label_counts": [np.bincount(y[i], minlength=10).tolist() for i in val_idx]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=".cache/mnist")
    p.add_argument("--output", type=Path, default=Path("experiments/results/b1_b2_lattice_noniid.json"))
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31, 43, 59])
    p.add_argument("--nodes", type=int, default=16)
    p.add_argument("--mode", choices=["iid", "non_iid"], default="non_iid")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--attack-start", type=int, default=10)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16])
    p.add_argument("--attackers", type=int, nargs="+", default=[0, 1, 5],
                   help="lattice 4x4: 0 corner (deg 2), 1 edge (deg 3), 5 interior (deg 4)")
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--validation", type=int, default=200)
    p.add_argument("--audit-per-class", type=int, default=100)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--defense", choices=DEFENSES, default="uniform")
    p.add_argument("--gate-tol", type=float, default=1.0, help="gate: accepted extra val-loss (nats)")
    p.add_argument("--trim-b", type=int, default=1)
    # BALANCE paper defaults (arXiv 2406.10416, Sec. 6): alpha=0.5, gamma=0.3, kappa=1, lambda(t)=t/T.
    p.add_argument("--balance-gamma", type=float, default=0.3)
    p.add_argument("--balance-kappa", type=float, default=1.0)
    p.add_argument("--balance-alpha", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    args = p.parse_args()

    shared = {k: v for k, v in vars(args).items() if k not in ("output", "workers", "cache")}
    jobs = []
    for seed in args.seeds:
        jobs.append({"seed": seed, "case": "local", "attacker": None, "lam": 0.0, "args": shared})
        jobs.append({"seed": seed, "case": "dfl", "attacker": None, "lam": 0.0, "args": shared})
        for att in args.attackers:
            for lam in args.lambdas:
                jobs.append({"seed": seed, "case": "dfl", "attacker": att, "lam": lam, "args": shared})
    start = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=(args.cache,)) as pool:
        results = pool.map(run, jobs)
    elapsed = time.time() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    args.output.write_text(json.dumps({
        "config": shared, "topology": "lattice", "neighbors": lattice(args.nodes),
        "source_sha256": hashlib.sha256(source).hexdigest(), "elapsed_s": round(elapsed, 1),
        "runs": results}))
    print(f"{len(jobs)} runs in {elapsed:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
