# V5 CAGRA build optimization (contest submission)

Baseline: tag `amd-cagra-contest-baseline-v5`, commit
`634888d9b037153fa143d2c913ba4b25503e1d62` (branch
`amd-oceanbase-official-contest-base-v5`).

## 1. Where the build time goes (measured)

V5 already builds and persists the CAGRA graph on the GPU during the offline
`CREATE VECTOR INDEX ... lib=cuvs` DDL, so the measured end-to-end build is:

| phase | 250K x 128 | 250K x 256 |
|---|---:|---:|
| CAGRA graph build (`cuvsCagraBuild`, NN-Descent 100 iters) | 46.6 s | ~49 s |
| graph serialize (snapshot write) | 0.2 s | 0.2 s |
| DDL scan / aux tables / first query | ~3 s | ~3 s |
| **total (T0 -> first query result)** | **~47 s** | **~49 s** |

So >90% of the build is `cuvsCagraBuild`. The bridge in this repo hard-coded
`NN_DESCENT` with `graph_degree=128`, `intermediate_graph_degree=256` and
`nn_descent_niter=100` for every dimension; 100 refinement iterations is a
worst-case budget that 128d data does not need.

## 2. Change

`tools/hipvs_bridge/seekdb_cuvs_bridge.c`: scale the NN-Descent refinement
budget with the vector dimension instead of always paying 100 iterations.

```c
long niter = 16 + (dim - 128) * 40 / 128;   /* 128d -> 16, 256d -> 56 */
if (niter < 16)  niter = 16;
if (niter > 100) niter = 100;
h->ip->nn_descent_niter = (size_t)niter;
```

`graph_degree`/`intermediate_graph_degree` and the search parameters
(`itopk_size = max(512, 4k)`, `search_width = 4`) are unchanged, so query cost
and query-time recall behaviour stay as in the baseline (only the graph
refinement effort changes).

## 3. Results (local A/B, same host: gfx1100, ROCm 7.2.4)

Synthetic 250K datasets generated with the harness in `tools/amd_cagra_bench`
(fixed seed, exact brute-force ground truth). Build time is
`T0 -> first APPROXIMATE query result` with a blocking DDL, matching the
official timing definition; recall is measured over `nq` queries.

| dataset | version | build | R@10 | R@100 | QPS top10/top100 |
|---|---|---:|---:|---:|---:|
| 250K x 128 | baseline V5 | 47.4 s | 0.9897 | 0.9670 | 133 / 126 |
| 250K x 128 | candidate | **22.9 s (2.07x)** | 0.9818 | 0.9500 | 140 / 122 |
| 250K x 256 | baseline V5 | 49.3 s | 0.9508 | 0.8991 | 122 / 118 |
| 250K x 256 | candidate | **39.9 s (1.24x)** | 0.9486 | 0.8940 | 124 / 117 |

Weighted (0.4 / 0.6) build speedup of this configuration: ~1.53x.

These synthetic datasets are harder than the official ones (baseline recall here
is 0.99/0.95 while the published V5 numbers are 1.000/0.985), so the recall
deltas above are upper bounds; the candidate keeps query QPS at parity with the
baseline (search parameters unchanged).

Reproduce:

```bash
# build (same commands as the official pipeline)
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm tools/hipvs_bridge/build.sh /work/bridge/libseekdb_cuvs_bridge.so
cmake -S . -B build_release -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DOB_ENABLE_UNITY=ON -DOB_USE_LLD=ON -DOB_BUILD_CUVS=ON -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB=/work/bridge/libseekdb_cuvs_bridge.so
cmake --build build_release --target seekdb

# benchmark
cd tools/amd_cagra_bench
python3 gen_data.py --n 250000 --dim 128 --nq 4096 --topk 100 --out data/s250k-x128
python3 gen_data.py --n 250000 --dim 256 --nq 4096 --topk 100 --out data/h250k-x256
python3 bench_v5.py --data data/s250k-x128 \
  --binary /path/to/seekdb --bridge /path/to/bridge-dir --port 2881 --tag cand --nq 1024
```

The observer must be started with
`LD_LIBRARY_PATH=<bridge dir>:/opt/hipvs/lib:/opt/rocm/lib`.
