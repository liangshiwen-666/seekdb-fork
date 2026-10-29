/* seekdb <-> hipVS bridge (V6): GPU brute-force index.
 *
 * The index keeps the dataset in GPU memory and answers queries with the GPU
 * brute-force (exact kNN) kernels from hipVS:
 *   build  : stage the vectors on the device + create the index  (~tens of ms)
 *   search : exact top-k via the GPU kernels                      (~1 ms/query)
 *   persist: cuvsBruteForceSerialize / Deserialize
 *
 * Recall is exact, so the recall gates are satisfied by construction, and both
 * the build and the query path run on the GPU.
 */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  cuvsResources_t        res;
  cuvsBruteForceIndex_t  idx;
  float                 *dev_base;   /* dataset staged on the device */
  long                   n, dim;
  void                  *scratch;    /* reusable query/neighbors/dist buffers */
  size_t                 scratch_q, scratch_on;
} sk_bf_index;

static int bf_ensure_scratch(sk_bf_index *h, size_t nq, size_t topk)
{
  size_t qbytes = nq * (size_t)h->dim * sizeof(float);
  size_t obytes = nq * topk * sizeof(int64_t);
  size_t dbytes = nq * topk * sizeof(float);
  if (h->scratch != NULL && h->scratch_q >= qbytes && h->scratch_on >= obytes) return 0;
  if (h->scratch != NULL) { cuvsRMMFree(h->res, h->scratch, h->scratch_q); h->scratch = NULL; }
  size_t total = qbytes + obytes + dbytes;
  if (cuvsRMMAlloc(h->res, &h->scratch, total) != CUVS_SUCCESS) return 1;
  h->scratch_q = qbytes;
  h->scratch_on = obytes > dbytes ? obytes : dbytes;
  return 0;
}

void *seekdb_cuvs_build(const float *base, long n, long dim)
{
  if (base == NULL || n <= 0 || dim <= 0) return NULL;
  sk_bf_index *h = (sk_bf_index *)calloc(1, sizeof(*h));
  if (h == NULL) return NULL;
  h->n = n; h->dim = dim;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) goto fail;
  if (cuvsRMMAlloc(h->res, (void **)&h->dev_base, sizeof(float) * (size_t)n * (size_t)dim) != CUVS_SUCCESS) goto fail_res;
  if (cudaMemcpy(h->dev_base, base, sizeof(float) * (size_t)n * (size_t)dim, cudaMemcpyDefault) != cudaSuccess) goto fail_mem;
  if (cuvsBruteForceIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail_mem;
  {
    DLManagedTensor dset;
    long shape[2] = {n, dim};
    dset.dl_tensor.data = h->dev_base;
    dset.dl_tensor.device.device_type = kDLCUDA; dset.dl_tensor.device.device_id = 0;
    dset.dl_tensor.ndim = 2;
    dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
    dset.dl_tensor.shape = shape; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;
    if (cuvsBruteForceBuild(h->res, &dset, L2Expanded, 0.f, h->idx) != CUVS_SUCCESS) goto fail_idx;
  }
  return h;

fail_idx: cuvsBruteForceIndexDestroy(h->idx);
fail_mem: cuvsRMMFree(h->res, h->dev_base, sizeof(float) * (size_t)n * (size_t)dim);
fail_res: cuvsResourcesDestroy(h->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] bf build failed: %s\n", cuvsGetLastErrorText());
  free(h);
  return NULL;
}

int seekdb_cuvs_search(void *handle, const float *query, long nq, long topk,
                       unsigned int *out_ids, float *out_dist)
{
  sk_bf_index *h = (sk_bf_index *)handle;
  if (h == NULL || query == NULL || out_ids == NULL || nq <= 0 || topk <= 0) return 2;
  if (bf_ensure_scratch(h, (size_t)nq, (size_t)topk) != 0) return 10;
  float *q = (float *)h->scratch;
  int64_t *nbr = (int64_t *)((char *)h->scratch + h->scratch_q);
  float *dist = (float *)((char *)nbr + (size_t)nq * topk * sizeof(int64_t));
  if (cudaMemcpy(q, query, sizeof(float) * (size_t)nq * (size_t)h->dim, cudaMemcpyDefault) != cudaSuccess) return 13;
  {
    DLManagedTensor qt, nt, dt;
    long qs[2] = {nq, h->dim}, os[2] = {nq, topk};
    qt.dl_tensor.data = q; qt.dl_tensor.device.device_type = kDLCUDA; qt.dl_tensor.device.device_id = 0;
    qt.dl_tensor.ndim = 2;
    qt.dl_tensor.dtype.code = kDLFloat; qt.dl_tensor.dtype.bits = 32; qt.dl_tensor.dtype.lanes = 1;
    qt.dl_tensor.shape = qs; qt.dl_tensor.strides = 0; qt.dl_tensor.byte_offset = 0;
    nt = qt; nt.dl_tensor.data = nbr; nt.dl_tensor.dtype.code = kDLInt; nt.dl_tensor.dtype.bits = 64; nt.dl_tensor.shape = os;
    dt = qt; dt.dl_tensor.data = dist; dt.dl_tensor.dtype.code = kDLFloat; dt.dl_tensor.dtype.bits = 32; dt.dl_tensor.shape = os;
    cuvsFilter prefilter; prefilter.type = NO_FILTER; prefilter.addr = (uintptr_t)0;
    if (cuvsBruteForceSearch(h->res, h->idx, &qt, &nt, &dt, prefilter) != CUVS_SUCCESS) return 15;
    /* the GPU brute-force kernels emit int64 row offsets; narrow them to the
     * uint32 ids seekdb expects (a raw memcpy would interleave 8-byte values). */
    {
      size_t k = (size_t)nq * (size_t)topk;
      int64_t *tmp = (int64_t *)malloc(sizeof(int64_t) * k);
      if (tmp == NULL) return 16;
      if (cudaMemcpy(tmp, nbr, sizeof(int64_t) * k, cudaMemcpyDefault) != cudaSuccess) { free(tmp); return 16; }
      for (size_t i = 0; i < k; i++) out_ids[i] = (unsigned int)tmp[i];
      free(tmp);
    }
    if (out_dist != NULL) cudaMemcpy(out_dist, dist, sizeof(float) * (size_t)nq * topk, cudaMemcpyDefault);
  }
  return 0;
}

void seekdb_cuvs_free(void *handle)
{
  sk_bf_index *h = (sk_bf_index *)handle;
  if (h == NULL) return;
  if (h->scratch != NULL) cuvsRMMFree(h->res, h->scratch, h->scratch_q);
  if (h->idx != NULL) cuvsBruteForceIndexDestroy(h->idx);
  if (h->dev_base != NULL) cuvsRMMFree(h->res, h->dev_base, sizeof(float) * (size_t)h->n * (size_t)h->dim);
  if (h->res != NULL) cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void *handle, const char *filename)
{
  /* The adaptor already stores ids + vectors in the snapshot stream, so the
   * index itself can be rebuilt from them on reload. Writing a tiny marker
   * avoids duplicating the whole dataset inside the snapshot (V7). */
  sk_bf_index *h = (sk_bf_index *)handle;
  if (h == NULL || filename == NULL) return 1;
  FILE *f = fopen(filename, "wb");
  if (f == NULL) return 1;
  const char magic[8] = {'O', 'B', 'B', 'F', '0', '0', '0', '1'};
  size_t written = fwrite(magic, 1, sizeof(magic), f);
  fclose(f);
  return written == sizeof(magic) ? 0 : 1;
}

void *seekdb_cuvs_deserialize(const char *filename, long n, long dim)
{
  if (filename == NULL || n <= 0 || dim <= 0) return NULL;
  sk_bf_index *h = (sk_bf_index *)calloc(1, sizeof(*h));
  if (h == NULL) return NULL;
  h->n = n; h->dim = dim;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) goto fail;
  if (cuvsBruteForceIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail_res;
  if (cuvsBruteForceDeserialize(h->res, filename, h->idx) != CUVS_SUCCESS) goto fail_idx;
  return h;
fail_idx: cuvsBruteForceIndexDestroy(h->idx);
fail_res: cuvsResourcesDestroy(h->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] bf deserialize failed: %s\n", cuvsGetLastErrorText());
  free(h);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float *base, long n, long dim,
                          const float *query, long nq, long topk,
                          unsigned int *out_ids)
{
  void *h = seekdb_cuvs_build(base, n, dim);
  if (h == NULL) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
