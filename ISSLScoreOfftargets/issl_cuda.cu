// ISSLScoreOfftargets/issl_cuda.cu
// GPU implementation for ISSL off-target scoring with CUDA acceleration
#include "issl_cuda.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdio>

// CUDA error checking macro for safe GPU operations
#define CUDA_OK(stmt)                                                        \
    do                                                                       \
    {                                                                        \
        cudaError_t _e = (stmt);                                             \
        if (_e != cudaSuccess)                                               \
        {                                                                    \
            std::fprintf(stderr, "CUDA error at %s:%d: %s -> %s\n",          \
                         __FILE__, __LINE__, #stmt, cudaGetErrorString(_e)); \
            throw std::runtime_error("CUDA failure");                        \
        }                                                                    \
    } while (0)

// Functor for comparing query-ID pairs during deduplication
struct KeyEqQid
{
    __host__ __device__ bool operator()(const thrust::tuple<int, uint32_t> &a,
                                        const thrust::tuple<int, uint32_t> &b) const
    {
        return thrust::get<0>(a) == thrust::get<0>(b) &&
               thrust::get<1>(a) == thrust::get<1>(b);
    }
};

// Functor for combining occurrence counts and mismatch data during deduplication
struct OccPlusKeepMism
{
    __host__ __device__
        thrust::tuple<uint32_t, uint64_t>
        operator()(const thrust::tuple<uint32_t, uint64_t> &A,
                   const thrust::tuple<uint32_t, uint64_t> &B) const
    {
        // Keep FIRST tuple's occ + mism (parity with CPU's first-hit semantics).
        const uint32_t occA = thrust::get<0>(A);
        const uint64_t mismA = thrust::get<1>(A);
        (void)B; // unused
        return thrust::make_tuple(occA, mismA);
    }
};

// SEQUENCE ENCODING 
// Convert DNA sequences to 64-bit signatures for efficient comparison

// Nucleotide lookup table: A=0, C=1, G=2, T=3, others=0
__device__ __forceinline__ uint8_t nuc_lut(char c)
{
    switch (c)
    {
    case 'C': return 1;
    case 'G': return 2;
    case 'T': return 3;
    default:  return 0;
    }
}

// GPU kernel to encode DNA sequences into 64-bit signatures
__global__ void k_encode(const char *buf, int stride, int seqlen, uint64_t *sigs, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const char *s = buf + i * stride;
    uint64_t sig = 0;
#pragma unroll
    for (int j = 0; j < seqlen; ++j)
    {
        sig |= (uint64_t)nuc_lut(s[j]) << (j * 2);
    }
    sigs[i] = sig;
}

// Host function to encode sequences on GPU
void gpu_encode_sequences(const char *h_buf, int n, int stride, int seqlen, uint64_t *h_out)
{
    char *d_buf = nullptr;
    uint64_t *d_out = nullptr;
    CUDA_OK(cudaMalloc(&d_buf, (size_t)n * stride));
    CUDA_OK(cudaMalloc(&d_out, n * sizeof(uint64_t)));
    CUDA_OK(cudaMemcpy(d_buf, h_buf, (size_t)n * stride, cudaMemcpyHostToDevice));
    dim3 blk(256), grd((n + blk.x - 1) / blk.x);
    k_encode<<<grd, blk>>>(d_buf, stride, seqlen, d_out, n);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(h_out, d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_buf));
    CUDA_OK(cudaFree(d_out));
}

// DISTANCE SCANNING 
// Find off-targets within maximum distance using slice-based approach
// Count hits per query for a single slice (first pass)
__global__ void k_distance_count_slice(
    const uint64_t *__restrict__ d_querySigs, int queryCount,
    const uint64_t *__restrict__ d_offtargets,
    const uint64_t *__restrict__ d_allSignatures,           // full table
    const size_t   *__restrict__ d_allSlicelistSizes_slice, // sizes for this slice only
    const uint32_t *__restrict__ d_prefixFlat_slice,        // prefix for this slice only
    const uint64_t *__restrict__ d_posIdx_slice,            // positions (len L)
    int L, size_t sliceBaseOffset, int maxDist,
    int * __restrict__ d_counts)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= queryCount) return;

    const uint64_t qsig = d_querySigs[q];

    // Build sub-code
    uint64_t sub = 0ULL;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        if (j >= L) break;
        const uint64_t p = d_posIdx_slice[j];
        sub |= ((qsig >> (p * 2)) & 3ULL) << (j * 2);
    }

    const size_t cnt   = d_allSlicelistSizes_slice[sub];
    const size_t begin = sliceBaseOffset + (size_t)d_prefixFlat_slice[sub];
    const uint64_t *ptr = d_allSignatures + begin;

    int local = 0;
    for (size_t t = 0; t < cnt; ++t) {
        const uint64_t packed = ptr[t];
        const uint32_t id  = (uint32_t)(packed & 0xFFFFFFFFULL);
        const uint64_t x = qsig ^ d_offtargets[id];
        const uint64_t mism = ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1) | (x & 0x5555555555555555ULL);
        const int dist = __popcll(mism);
        if (dist <= maxDist) ++local;
    }
    d_counts[q] = local;
}

// Emit hits per query for a single slice (second pass)
__global__ void k_distance_emit_slice(
    const uint64_t *__restrict__ d_querySigs, int queryCount,
    const uint64_t *__restrict__ d_offtargets,
    const uint64_t *__restrict__ d_allSignatures,
    const size_t   *__restrict__ d_allSlicelistSizes_slice,
    const uint32_t *__restrict__ d_prefixFlat_slice,
    const uint64_t *__restrict__ d_posIdx_slice,
    int L, size_t sliceBaseOffset, int maxDist,
    const size_t   *__restrict__ d_offsets,   // Q+1
    int baseQ,
    int4 * __restrict__ d_hits,
    uint64_t * __restrict__ d_mismatches)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= queryCount) return;

    const uint64_t qsig = d_querySigs[q];

    // Build sub-code
    uint64_t sub = 0ULL;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        if (j >= L) break;
        const uint64_t p = d_posIdx_slice[j];
        sub |= ((qsig >> (p * 2)) & 3ULL) << (j * 2);
    }

    const size_t cnt   = d_allSlicelistSizes_slice[sub];
    const size_t begin = sliceBaseOffset + (size_t)d_prefixFlat_slice[sub];
    const uint64_t *ptr = d_allSignatures + begin;

    size_t out = d_offsets[q];
    for (size_t t = 0; t < cnt; ++t) {
        const uint64_t packed = ptr[t];
        const uint32_t id  = (uint32_t)(packed & 0xFFFFFFFFULL);
        const uint32_t occ = (uint32_t)(packed >> 32);

        const uint64_t x = qsig ^ d_offtargets[id];
        const uint64_t mism = ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1) | (x & 0x5555555555555555ULL);
        const int dist = __popcll(mism);
        if (dist <= maxDist) {
            d_hits[out] = make_int4(baseQ + q, (int)id, (int)occ, (int)(mism >> 32));
            d_mismatches[out] = mism;
            ++out;
        }
    }
}

// Main distance scanning function using slice-based approach
void gpu_distance_scan_by_slice_buffered(
    const std::vector<uint64_t>& querySigs,
    const std::vector<uint64_t>& offtargets,
    const std::vector<uint64_t>& allSignatures,   // packed [occ:32|id:32]
    const std::vector<size_t>&   allSlicelistSizes,
    const std::vector<int>&      sliceLen,
    const std::vector<size_t>&   sliceSizesOffset,
    const std::vector<size_t>&   sliceBaseOffset,
    const std::vector<uint32_t>& prefixFlat,
    const std::vector<size_t>&   prefixOffset,
    const std::vector<uint64_t>& posIdxFlat,
    const std::vector<size_t>&   posOffset,
    int maxDist,
    std::vector<Hit>& out_hits)
{
    const int Q = (int)querySigs.size();
    // N not used here
    const int S = (int)sliceLen.size();

    // Upload static data once
    uint64_t *d_off=nullptr, *dSig=nullptr, *dPos=nullptr;
    size_t *dSizes=nullptr, *dSizesOff=nullptr, *dBaseOff=nullptr, *dPrefOff=nullptr, *dPosOff=nullptr;
    int *dLen=nullptr;
    uint32_t *dPref=nullptr;

    CUDA_OK(cudaMalloc(&d_off, offtargets.size() * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&dSig, allSignatures.size() * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&dSizes, allSlicelistSizes.size() * sizeof(size_t)));
    CUDA_OK(cudaMalloc(&dLen, sliceLen.size() * sizeof(int)));
    CUDA_OK(cudaMalloc(&dSizesOff, sliceSizesOffset.size() * sizeof(size_t)));
    CUDA_OK(cudaMalloc(&dBaseOff,  sliceBaseOffset.size()  * sizeof(size_t)));
    CUDA_OK(cudaMalloc(&dPref,     prefixFlat.size()       * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&dPrefOff,  prefixOffset.size()     * sizeof(size_t)));
    CUDA_OK(cudaMalloc(&dPos,      posIdxFlat.size()       * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&dPosOff,   posOffset.size()        * sizeof(size_t)));

    CUDA_OK(cudaMemcpy(d_off,  offtargets.data(), offtargets.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dSig,   allSignatures.data(), allSignatures.size()*sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dSizes, allSlicelistSizes.data(), allSlicelistSizes.size()*sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dLen,   sliceLen.data(), sliceLen.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dSizesOff, sliceSizesOffset.data(), sliceSizesOffset.size()*sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dBaseOff,  sliceBaseOffset.data(),  sliceBaseOffset.size() *sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dPref,     prefixFlat.data(),       prefixFlat.size()      *sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dPrefOff,  prefixOffset.data(),     prefixOffset.size()    *sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dPos,      posIdxFlat.data(),       posIdxFlat.size()      *sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dPosOff,   posOffset.data(),        posOffset.size()       *sizeof(size_t), cudaMemcpyHostToDevice));

    // Upload queries once
    uint64_t* d_q = nullptr;
    CUDA_OK(cudaMalloc(&d_q, Q * sizeof(uint64_t)));
    CUDA_OK(cudaMemcpy(d_q, querySigs.data(), Q * sizeof(uint64_t), cudaMemcpyHostToDevice));

    out_hits.clear();
    out_hits.reserve(Q * 128);

    const int threads = 256;
    const dim3 blk(threads), grd((Q + threads - 1) / threads);

    for (int s = 0; s < S; ++s) {
        const int   L = sliceLen[s];
        const size_t sizesBase  = sliceSizesOffset[s];
        const size_t prefixBase = prefixOffset[s];
        const size_t baseSigOff = sliceBaseOffset[s];
        const size_t posBase    = posOffset[s];

        // Slice-local pointers
        const size_t*   d_sizes_slice   = dSizes + sizesBase;
        const uint32_t* d_prefix_slice  = dPref  + prefixBase;
        const uint64_t* d_pos_slice     = dPos   + posBase;

        // PASS 1: count per query
        thrust::device_vector<int> d_counts(Q, 0);
        k_distance_count_slice<<<grd, blk>>>(
            d_q, Q, d_off, dSig,
            d_sizes_slice, d_prefix_slice,
            d_pos_slice, L, baseSigOff, maxDist,
            thrust::raw_pointer_cast(d_counts.data()));
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaGetLastError());

        // Exclusive scan -> offsets (Q+1), compute total
        thrust::device_vector<size_t> d_offsets(Q + 1);
        d_offsets[0] = 0;
        thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin());

        size_t lastOffset = 0, lastCount = 0;
        CUDA_OK(cudaMemcpy(&lastOffset,
                           thrust::raw_pointer_cast(d_offsets.data()) + (Q - 1),
                           sizeof(size_t), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(&lastCount,
                           thrust::raw_pointer_cast(d_counts.data()) + (Q - 1),
                           sizeof(int), cudaMemcpyDeviceToHost));
        const size_t totalHits = lastOffset + lastCount;

        // Edge case: no hits for this slice
        if (totalHits == 0) {
            // Still set d_offsets[Q] = 0 for completeness
            size_t zero = 0;
            CUDA_OK(cudaMemcpy(thrust::raw_pointer_cast(d_offsets.data()) + Q,
                               &zero, sizeof(size_t), cudaMemcpyHostToDevice));
            continue;
        } else {
            CUDA_OK(cudaMemcpy(thrust::raw_pointer_cast(d_offsets.data()) + Q,
                               &totalHits, sizeof(size_t), cudaMemcpyHostToDevice));
        }

        // Allocate exact buffers and EMIT
        int4* d_hits = nullptr;
        uint64_t* d_mism = nullptr;
        CUDA_OK(cudaMalloc(&d_hits, totalHits * sizeof(int4)));
        CUDA_OK(cudaMalloc(&d_mism, totalHits * sizeof(uint64_t)));

        k_distance_emit_slice<<<grd, blk>>>(
            d_q, Q, d_off, dSig,
            d_sizes_slice, d_prefix_slice,
            d_pos_slice, L, baseSigOff, maxDist,
            thrust::raw_pointer_cast(d_offsets.data()),
            /*baseQ*/ 0,
            d_hits, d_mism);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaGetLastError());

        // Copy back and append
        std::vector<int4>      h_hits(totalHits);
        std::vector<uint64_t>  h_mm(totalHits);
        CUDA_OK(cudaMemcpy(h_hits.data(), d_hits, totalHits * sizeof(int4), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(h_mm.data(),   d_mism, totalHits * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        const size_t old = out_hits.size();
        out_hits.resize(old + totalHits);
        for (size_t i = 0; i < totalHits; ++i) {
            out_hits[old + i] = { h_hits[i].x, (uint32_t)h_hits[i].y, (uint32_t)h_hits[i].z, h_mm[i] };
        }

        CUDA_OK(cudaFree(d_hits));
        CUDA_OK(cudaFree(d_mism));
    }

    // Cleanup static
    CUDA_OK(cudaFree(d_q));
    CUDA_OK(cudaFree(d_off));
    CUDA_OK(cudaFree(dSig));
    CUDA_OK(cudaFree(dSizes));
    CUDA_OK(cudaFree(dLen));
    CUDA_OK(cudaFree(dSizesOff));
    CUDA_OK(cudaFree(dBaseOff));
    CUDA_OK(cudaFree(dPref));
    CUDA_OK(cudaFree(dPrefOff));
    CUDA_OK(cudaFree(dPos));
    CUDA_OK(cudaFree(dPosOff));
}

// DEDUPLICATION 
// Remove duplicate (query, off-target) pairs and build query offsets

DedupResult gpu_dedup_by_qid(const std::vector<Hit> &hits, int Q)
{
    using thrust::device_vector;
    using thrust::host_vector;
    using thrust::make_tuple;
    using thrust::make_zip_iterator;

    DedupResult res;
    if (hits.empty()) {
        res.q_u.clear();
        res.id_u.clear();
        res.occ_u.clear();
        res.mism_u.clear();
        res.qOffset.assign(Q + 1, 0);
        res.distinctCount = 0;
        return res;
    }

    const size_t H = hits.size();

    // Build on host, single bulk copies to device
    host_vector<int>       h_q(H);
    host_vector<uint32_t>  h_id(H), h_occ(H);
    host_vector<uint64_t>  h_mism(H);

    for (size_t i = 0; i < H; ++i) {
        h_q[i]   = hits[i].q;
        h_id[i]  = hits[i].id;
        h_occ[i] = hits[i].occ;
        h_mism[i]= hits[i].mismatches;
    }

    device_vector<int>       d_q   = h_q;
    device_vector<uint32_t>  d_id  = h_id;
    device_vector<uint32_t>  d_occ = h_occ;
    device_vector<uint64_t>  d_mism= h_mism;

    // sort by (q,id)
    auto keys_begin = make_zip_iterator(make_tuple(d_q.begin(), d_id.begin()));
    auto keys_end   = make_zip_iterator(make_tuple(d_q.end(),   d_id.end()));
    auto vals_begin = make_zip_iterator(make_tuple(d_occ.begin(), d_mism.begin()));
    thrust::sort_by_key(keys_begin, keys_end, vals_begin);

    // reduce_by_key to dedup (q,id), keep FIRST (parity with CPU)
    device_vector<int>       q_u(d_q.size());
    device_vector<uint32_t>  id_u(d_id.size());
    device_vector<uint32_t>  occ_u(d_occ.size());
    device_vector<uint64_t>  mism_u(d_mism.size());

    auto out_keys_begin = make_zip_iterator(make_tuple(q_u.begin(), id_u.begin()));
    auto out_vals_begin = make_zip_iterator(make_tuple(occ_u.begin(), mism_u.begin()));

    auto new_end = thrust::reduce_by_key(
        keys_begin, keys_end, vals_begin,
        out_keys_begin, out_vals_begin,
        KeyEqQid{}, OccPlusKeepMism{});

    size_t U = new_end.first - out_keys_begin;
    q_u.resize(U); id_u.resize(U); occ_u.resize(U); mism_u.resize(U);

    // Build qOffset (dense, size Q+1)
    device_vector<int> q_unique(U);
    device_vector<int> q_counts_compact(U);
    auto end_counts = thrust::reduce_by_key(
        q_u.begin(), q_u.end(),
        thrust::make_constant_iterator<int>(1),
        q_unique.begin(),
        q_counts_compact.begin());
    size_t K = end_counts.first - q_unique.begin();
    q_unique.resize(K);
    q_counts_compact.resize(K);

    device_vector<int> q_counts(Q, 0);
    thrust::scatter(q_counts_compact.begin(), q_counts_compact.end(),
                    q_unique.begin(), q_counts.begin());

    device_vector<size_t> qOffset(Q + 1);
    qOffset[0] = 0;
    thrust::inclusive_scan(q_counts.begin(), q_counts.end(), qOffset.begin() + 1);

    // Distinct IDs across ALL queries
    device_vector<uint32_t> id_copy = id_u;
    thrust::sort(id_copy.begin(), id_copy.end());
    auto id_end = thrust::unique(id_copy.begin(), id_copy.end());
    uint64_t distinct = static_cast<uint64_t>(id_end - id_copy.begin());

    // copy out to host
    res.q_u.resize(U);        thrust::copy(q_u.begin(),     q_u.end(),     res.q_u.begin());
    res.id_u.resize(U);       thrust::copy(id_u.begin(),    id_u.end(),    res.id_u.begin());
    res.occ_u.resize(U);      thrust::copy(occ_u.begin(),   occ_u.end(),   res.occ_u.begin());
    res.mism_u.resize(U);     thrust::copy(mism_u.begin(),  mism_u.end(),  res.mism_u.begin());
    res.qOffset.resize(Q + 1);thrust::copy(qOffset.begin(), qOffset.end(), res.qOffset.begin());
    res.distinctCount = distinct;

    return res;
}

// SCORING 
// Calculate MIT and CFD scores for off-targets

// CFD penalty tables stored in constant memory for fast access
__constant__ double d_cfdPam[16];
__constant__ double d_cfdPos[/* 20*16 or more; will load exact size at runtime */ 1024]; // large enough guard

// Load CFD penalty tables into GPU constant memory
void gpu_load_cfd_tables(const double *pam, size_t pamSize, const double *pos, size_t posSize)
{
    if (pamSize > 16)
        throw std::runtime_error("cfdPamPenalties size > 16 unexpected");
    CUDA_OK(cudaMemcpyToSymbol(d_cfdPam, pam, pamSize * sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpyToSymbol(d_cfdPos, pos, posSize * sizeof(double), 0, cudaMemcpyHostToDevice));
}

// Helper function for population count
__device__ __forceinline__ int popc64(uint64_t x) { return __popcll(x); }

// Binary search for MIT score lookup
__device__ double lookupMIT(uint64_t mism, const uint64_t *masks, const double *vals, int N)
{
    int lo = 0, hi = N - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) >> 1;
        uint64_t m = masks[mid];
        if (m == mism)
            return vals[mid];
        if (m < mism)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return 0.0;
}

// Main scoring kernel: one block per query
// scoreMethod: 0=and,1=or,2=avg,3=mit,4=cf
__global__ void k_score_queries(
    const uint64_t *__restrict__ querySig,   // Q
    const uint64_t *__restrict__ offtargets, // N
    const int *__restrict__ q_u,             // U
    const uint32_t *__restrict__ id_u,       // U
    const uint32_t *__restrict__ occ_u,      // U
    const uint64_t *__restrict__ mism_u,     // U
    const size_t *__restrict__ qOffset,      // Q+1
    const uint64_t *__restrict__ mitMasks,   // M
    const double *__restrict__ mitVals,      // M
    int M, double threshold, int scoreMethod,
    int calcMit, int calcCfd,
    double *__restrict__ outMIT, double *__restrict__ outCFD,
    uint64_t *__restrict__ outCount)
{
    int q = blockIdx.x;
    uint64_t searchSig = querySig[q];
    size_t begin = qOffset[q], end = qOffset[q + 1];

    double totMit = 0.0, totCfd = 0.0;
    uint64_t localCount = 0;
    double maximum_sum = (10000.0 - threshold * 100.0) / threshold;

    for (size_t i = begin; i < end; ++i)
    {
        uint32_t id = id_u[i];
        uint32_t occ = occ_u[i];
        uint64_t mm = mism_u[i];

        int dist = popc64(mm);
        if (dist > 4)
            continue;

        localCount += occ;

        // CFD
        if (calcCfd)
        {
            double cfd = 0.0;
            if (dist == 0)
            {
                cfd = 1.0;
            }
            else
            {
                cfd = d_cfdPam[0b1010]; // PAM NGG (matches CPU)
#pragma unroll
                for (int pos = 0; pos < 20; ++pos)
                {
                    uint64_t s = (searchSig >> (pos * 2)) & 3ULL;
                    uint64_t o = (offtargets[id] >> (pos * 2)) & 3ULL;
                    unsigned mask = (unsigned)((pos << 4) | (uint32_t)(s << 2) | (uint32_t)(o ^ 3U));
                    if (s != o)
                        cfd *= d_cfdPos[mask];
                }
            }
            totCfd += cfd * (double)occ;
        }

        // MIT via LUT
        if (calcMit && dist > 0)
        {
            double mit = lookupMIT(mm, mitMasks, mitVals, M);
            totMit += mit * (double)occ;
        }

        // early exit (mirror CPU)
        if (scoreMethod == 0 /*and*/)
        {
            if (totMit > maximum_sum && totCfd > maximum_sum)
                break;
        }
        else if (scoreMethod == 1 /*or */)
        {
            if (totMit > maximum_sum || totCfd > maximum_sum)
                break;
        }
        else if (scoreMethod == 2 /*avg*/)
        {
            if (((totMit + totCfd) / 2.0) > maximum_sum)
                break;
        }
        else if (scoreMethod == 3 /*mit*/)
        {
            if (totMit > maximum_sum)
                break;
        }
        else if (scoreMethod == 4 /*cfd*/)
        {
            if (totCfd > maximum_sum)
                break;
        }
    }

    outCount[q] = localCount;
    outMIT[q] = 10000.0 / (100.0 + totMit);
    outCFD[q] = 10000.0 / (100.0 + totCfd);
}

// Host function for GPU scoring with early exit optimization
void gpu_score_queries(
    const std::vector<uint64_t> &querySigs,
    const std::vector<uint64_t> &offtargets,
    const std::vector<int> &q_u,
    const std::vector<uint32_t> &id_u,
    const std::vector<uint32_t> &occ_u,
    const std::vector<uint64_t> &mism_u,
    const std::vector<size_t> &qOffset,
    const std::vector<uint64_t> &mitMasks,
    const std::vector<double> &mitVals,
    double threshold, int scoreMethod,
    bool calcMit, bool calcCfd,
    std::vector<double> &outMIT, std::vector<double> &outCFD, std::vector<uint64_t> &outCount)
{
    const int Q = (int)querySigs.size();
    const int U = (int)q_u.size();
    const int N = (int)offtargets.size();
    const int M = (int)mitMasks.size();

    // allocate device
    uint64_t *d_query = nullptr, *d_off = nullptr;
    int *d_q_u = nullptr;
    uint32_t *d_id_u = nullptr, *d_occ_u = nullptr;
    uint64_t *d_mism_u = nullptr;
    size_t *d_qOffset = nullptr;
    uint64_t *d_mitMasks = nullptr;
    double *d_mitVals = nullptr;
    double *d_outMIT = nullptr, *d_outCFD = nullptr;
    uint64_t *d_outCount = nullptr;

    CUDA_OK(cudaMalloc(&d_query, Q * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_off, N * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_q_u, U * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_id_u, U * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_occ_u, U * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_mism_u, U * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_qOffset, (Q + 1) * sizeof(size_t)));
    if (M > 0)
    {
        CUDA_OK(cudaMalloc(&d_mitMasks, M * sizeof(uint64_t)));
        CUDA_OK(cudaMalloc(&d_mitVals, M * sizeof(double)));
    }
    CUDA_OK(cudaMalloc(&d_outMIT, Q * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_outCFD, Q * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_outCount, Q * sizeof(uint64_t)));

    // copy
    CUDA_OK(cudaMemcpy(d_query, querySigs.data(), Q * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_off, offtargets.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_q_u, q_u.data(), U * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_id_u, id_u.data(), U * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_occ_u, occ_u.data(), U * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_mism_u, mism_u.data(), U * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_qOffset, qOffset.data(), (Q + 1) * sizeof(size_t), cudaMemcpyHostToDevice));
    if (M > 0)
    {
        CUDA_OK(cudaMemcpy(d_mitMasks, mitMasks.data(), M * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_mitVals, mitVals.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    }

    // launch (one block per query; 1 thread is fine since each query iterates its [begin,end))
    dim3 grd(Q), blk(1);
    k_score_queries<<<grd, blk>>>(
        d_query, d_off, d_q_u, d_id_u, d_occ_u, d_mism_u, d_qOffset,
        d_mitMasks, d_mitVals, M, threshold, scoreMethod,
        calcMit ? 1 : 0, calcCfd ? 1 : 0,
        d_outMIT, d_outCFD, d_outCount);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGetLastError());

    // copy back
    outMIT.resize(Q);
    outCFD.resize(Q);
    outCount.resize(Q);
    CUDA_OK(cudaMemcpy(outMIT.data(), d_outMIT, Q * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(outCFD.data(), d_outCFD, Q * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(outCount.data(), d_outCount, Q * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // free
    CUDA_OK(cudaFree(d_query));
    CUDA_OK(cudaFree(d_off));
    CUDA_OK(cudaFree(d_q_u));
    CUDA_OK(cudaFree(d_id_u));
    CUDA_OK(cudaFree(d_occ_u));
    CUDA_OK(cudaFree(d_mism_u));
    CUDA_OK(cudaFree(d_qOffset));
    if (M > 0)
    {
        CUDA_OK(cudaFree(d_mitMasks));
        CUDA_OK(cudaFree(d_mitVals));
    }
    CUDA_OK(cudaFree(d_outMIT));
    CUDA_OK(cudaFree(d_outCFD));
    CUDA_OK(cudaFree(d_outCount));
}
