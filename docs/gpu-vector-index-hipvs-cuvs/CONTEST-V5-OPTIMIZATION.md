# V5 CAGRA build optimization (contest submission)

Baseline: tag `amd-cagra-contest-baseline-v5`, commit
`634888d9b037153fa143d2c913ba4b25503e1d62`.

Two candidate versions are kept on top of the frozen V5 baseline:

| version | branch | change | local weighted build speedup |
|---|---|---|---|
| V0 (fallback) | `amd-cagra-submit-v0` | `niter = 32 + (dim-128)*40/128` (128d:32, 256d:72) | ~1.6x |
| **V1 (this branch)** | `amd-cagra-submit` | `niter = 20 + (dim-128)*36/128` (128d:20, 256d:56) | **~1.87x** |

## 1. Where the build time goes

V5 builds and persists the GPU CAGRA graph inside the offline
`CREATE VECTOR INDEX ... lib=cuvs` path. Measured with an instrumented bridge on
250K x 128 (gfx1100, ROCm 7.2.4):

| phase | time |
|---|---:|
| `cuvsCagraBuild` (NN-Descent, 100 iters) | 46.6 s |
| graph serialize (snapshot) | 0.2 s |
| DDL scan / aux tables / first query | ~3 s |

More than 90% of the measured build is `cuvsCagraBuild`, and the baseline bridge
uses a fixed 100-iteration NN-Descent budget for every dimension.

## 2. Change

`tools/hipvs_bridge/seekdb_cuvs_bridge.c`: scale the refinement budget with the
vector dimension (128d data reaches the recall plateau much earlier than 256d):

```c
long niter = 20 + (dim - 128) * 36 / 128;   /* 128d -> 20, 256d -> 56 */
if (niter < 20)  niter = 20;
if (niter > 100) niter = 100;
h->ip->nn_descent_niter = (size_t)niter;
```

`graph_degree` / `intermediate_graph_degree` and the search parameters
(`itopk = max(512, 4k)`, `search_width = 4`) are unchanged, so query cost stays
at baseline parity (measured QPS ratio 0.97-1.00).

Rejected alternatives (measured, no win): float16 dataset build (95.6s -> 93.4s,
no speedup), build-subset + `cuvsCagraExtend` (66.9s vs 63.6s full, slower),
IVF-PQ build (6.7s but recall 0.77/0.71 vs 0.99/0.97), and raising graph degree
to compensate fewer iterations (build cost grows back).

## 3. Measured A/B (same host, alternating, fresh observer per run)

Baseline = frozen V5 bridge, candidate = this bridge. Build is
`T0 -> first APPROXIMATE query result` with a blocking DDL; recall over 1024
queries; QPS over the same 1024 queries.

| dataset | version | build | speedup | R@10 | R@100 | QPS top10/top100 |
|---|---|---:|---:|---:|---:|---:|
| 250K x 128 | V5 baseline | 47.58 s | 1.00x | 0.9898 | 0.9661 | 136.5 / 123.2 |
| 250K x 128 | **V1** | **17.09 s** | **2.78x** | 0.9672 | 0.9317 | 132.3 / 123.4 |
| 250K x 256 | V5 baseline | 47.96 s | 1.00x | 0.9521 | 0.8974 | 126.9 / 119.3 |
| 250K x 256 | **V1** | **33.42 s** | **1.44x** | 0.9413 | 0.8897 | 125.0 / 115.0 |

Weighted build speedup (0.4 / 0.6): **~1.87x** (log-scaled build score ~8/14).

The synthetic datasets above are harder than the official ones (baseline recall
0.99/0.95 here vs 1.000/0.985 published for SIFT250K/Hidden250K), so the recall
deltas are upper bounds. Query QPS is at baseline parity, so the query gate
(>=90%) is satisfied; query performance points are not expected.

If the Hidden250K recall gate turns out to be too tight, switching the 256d
budget back to 72-88 iterations costs ~1-2 build points but restores recall to
near-baseline (the V0 branch is one such safer configuration).

## 4. Reproduce

```bash
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm tools/hipvs_bridge/build.sh /work/bridge/libseekdb_cuvs_bridge.so
cmake -S . -B build_release -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DOB_ENABLE_UNITY=ON -DOB_USE_LLD=ON -DOB_BUILD_CUVS=ON -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB=/work/bridge/libseekdb_cuvs_bridge.so
cmake --build build_release --target seekdb -j96

cd tools/amd_cagra_bench
python3 gen_data.py --n 250000 --dim 128 --nq 4096 --topk 100 --out data/s250k-x128
python3 gen_data.py --n 250000 --dim 256 --nq 4096 --topk 100 --out data/h250k-x256
python3 bench_v5.py --data data/s250k-x128 --binary /path/to/seekdb \
  --bridge /path/to/bridge --port 2881 --tag v1 --nq 1024
```
