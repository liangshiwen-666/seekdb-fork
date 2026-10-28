#!/usr/bin/env python3
"""gen_data.py -- reproducible synthetic L2 vector benchmark data.

Produces, for a fixed seed and shape:
  base.f32      (n, dim)  float32, row-major (C order)
  query.f32     (nq, dim) float32, row-major
  gt_top100.i32 (nq, 100) int32, exact brute-force L2 top-100 neighbour ids
  gt_dist.f32   (nq, 100) float32, exact distances matching gt ids
  meta.json     shape/seed/dtype/hashes, so a run can be proven to use the same data

Vectors are i.i.d. standard normal, drawn with numpy's PCG64 from a fixed seed, so the
output is byte-reproducible on any machine with the same numpy dtype semantics.

Ground truth is computed exactly (no approximation) with a blocked float64 GEMM pass;
memory stays bounded no matter how large --n is.

Examples:
  ./gen_data.py --n 100000  --dim 128 --nq 100 --out data/100k-x128
  ./gen_data.py --n 1000000 --dim 128 --nq 100 --out data/1m-x128
  ./gen_data.py --n 100000  --dim 768 --nq 100 --out data/100k-x768
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

VERSION = "1.0"


def sha256_file(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def exact_topk(base: np.ndarray, queries: np.ndarray, k: int, block: int = 4096):
    """Exact L2 top-k ids/distances, blocked float64 GEMM.

    Returns (ids int32 [nq,k], dists float32 [nq,k]) sorted by distance ascending.
    """
    n, _ = base.shape
    nq = queries.shape[0]
    k = min(k, n)
    q64 = queries.astype(np.float64, copy=False)
    qn = np.einsum("ij,ij->i", q64, q64)

    best_d = np.full((nq, k), np.inf, dtype=np.float64)
    best_i = np.full((nq, k), -1, dtype=np.int64)

    for start in range(0, n, block):
        blk = base[start:start + block]
        b64 = blk.astype(np.float64)
        bn = np.einsum("ij,ij->i", b64, b64)
        # d[i, j] = |q_i|^2 + |b_j|^2 - 2 q_i . b_j  (monotonic in true L2)
        d = qn[:, None] + bn[None, :] - 2.0 * (q64 @ b64.T)
        take = min(k, d.shape[1])
        part = np.argpartition(d, take - 1, axis=1)[:, :take]
        part_d = np.take_along_axis(d, part, axis=1)
        cand_d = np.concatenate([best_d, part_d], axis=1)
        cand_i = np.concatenate([best_i, part + start], axis=1)
        order = np.argsort(cand_d, axis=1)[:, :k]
        best_d = np.take_along_axis(cand_d, order, axis=1)
        best_i = np.take_along_axis(cand_i, order, axis=1)

    # exact distances for the winner ids (avoid the |b|^2 + |q|^2 - 2qb rounding)
    out_d = np.empty((nq, k), dtype=np.float32)
    for i in range(nq):
        diff = base[best_i[i]].astype(np.float64) - q64[i]
        out_d[i] = np.sqrt(np.einsum("ij,ij->i", diff, diff)).astype(np.float32)
    return best_i.astype(np.int32), out_d


def verify_sample(base, queries, ids, dists, samples=3, rng=None):
    """Brute-force re-check a few queries to prove the blocked GT is exact."""
    n = base.shape[0]
    for i in range(min(samples, queries.shape[0])):
        d = np.sqrt(((base - queries[i]) ** 2).sum(axis=1, dtype=np.float64))
        truth = np.sort(np.argsort(d)[: ids.shape[1]])
        got = np.sort(ids[i])
        if not np.array_equal(truth, got):
            raise SystemExit(f"GT self-check failed for query {i}")
        if not np.allclose(np.sort(d[got]), np.sort(dists[i].astype(np.float64)), atol=1e-4):
            raise SystemExit(f"GT distance self-check failed for query {i}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="number of base vectors")
    ap.add_argument("--dim", type=int, required=True, help="vector dimension")
    ap.add_argument("--nq", type=int, default=100, help="number of query vectors (default 100)")
    ap.add_argument("--topk", type=int, default=100, help="ground-truth depth (default 100)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=20261028, help="RNG seed (default 20261028)")
    ap.add_argument("--block", type=int, default=4096, help="GEMM block rows (default 4096)")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--no-verify", action="store_true", help="skip brute-force GT self-check")
    args = ap.parse_args()

    if args.n <= 0 or args.dim <= 0 or args.nq <= 0:
        print("n/dim/nq must be positive", file=sys.stderr)
        return 2
    if args.topk > args.n:
        print(f"topk {args.topk} > n {args.n}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_p = out / "base.f32"
    query_p = out / "query.f32"
    gt_p = out / f"gt_{args.nq}x{args.topk}.i32"
    gt_dist_p = out / "gt_dist.f32"
    meta_p = out / "meta.json"

    if not args.force:
        for p in (base_p, query_p, gt_p, meta_p):
            if p.exists():
                print(f"{p} exists; use --force to regenerate", file=sys.stderr)
                return 2

    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    print(f"[gen_data] n={args.n} dim={args.dim} nq={args.nq} topk={args.topk} seed={args.seed}",
          flush=True)

    base = rng.standard_normal((args.n, args.dim), dtype=np.float32)
    queries = rng.standard_normal((args.nq, args.dim), dtype=np.float32)
    base.tofile(base_p)
    queries.tofile(query_p)
    print(f"[gen_data] wrote base.f32 ({base.nbytes/2**20:.1f} MiB), query.f32", flush=True)

    t_gt = time.time()
    ids, dists = exact_topk(base, queries, args.topk, block=args.block)
    ids.tofile(gt_p)
    dists.tofile(gt_dist_p)
    print(f"[gen_data] ground truth top-{args.topk} in {time.time()-t_gt:.1f}s", flush=True)

    if not args.no_verify:
        verify_sample(base, queries, ids, dists)
        print("[gen_data] GT self-check OK (brute force + distance)", flush=True)

    meta = {
        "version": VERSION,
        "n": args.n,
        "dim": args.dim,
        "nq": args.nq,
        "topk": args.topk,
        "seed": args.seed,
        "dtype": "float32",
        "layout": "row-major (C order), little-endian",
        "distribution": "i.i.d. standard normal N(0,1)",
        "distance": "L2",
        "generator": "numpy.random.Generator(PCG64).standard_normal(dtype=float32)",
        "files": {
            "base": {"name": base_p.name, "bytes": base_p.stat().st_size,
                     "shape": [args.n, args.dim], "sha256": sha256_file(base_p)},
            "query": {"name": query_p.name, "bytes": query_p.stat().st_size,
                      "shape": [args.nq, args.dim], "sha256": sha256_file(query_p)},
            "gt": {"name": gt_p.name, "bytes": gt_p.stat().st_size,
                   "shape": [args.nq, args.topk], "sha256": sha256_file(gt_p)},
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gen_seconds": round(time.time() - t_start, 1),
    }
    meta_p.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[gen_data] done in {time.time()-t_start:.1f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
