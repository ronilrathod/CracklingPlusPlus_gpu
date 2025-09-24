// ISSLScoreOfftargets/issl_cuda.cu
#include "issl_cuda.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

// Concrete functors to avoid generic device lambdas
struct KeyEqQid
{
    __host__ __device__ bool operator()(const thrust::tuple<int, uint32_t> &a,
                                        const thrust::tuple<int, uint32_t> &b) const
    {
        return thrust::get<0>(a) == thrust::get<0>(b) &&
               thrust::get<1>(a) == thrust::get<1>(b);
    }
};

struct OccPlusKeepMism
{
    __host__ __device__
        thrust::tuple<uint32_t, uint64_t>
        operator()(const thrust::tuple<uint32_t, uint64_t> &A,
                   const thrust::tuple<uint32_t, uint64_t> &B) const
    {
        // Keep FIRST tuple’s occ + mism (parity with CPU's first-hit semantics).
        // If you want to be extra safe, swap the next line for max(occA, occB).
        const uint32_t occA = thrust::get<0>(A);
        const uint64_t mismA = thrust::get<1>(A);
        (void)B; // unused
        return thrust::make_tuple(occA, mismA);
    }
};

// --------------------------- 1) popcount sanity ----------------------------
__global__ void k_hamming(const uint64_t *a, const uint64_t *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __popcll(a[i] ^ b[i]);
}

void gpu_popcount_hamming(const uint64_t *h_a, const uint64_t *h_b, int n, int *h_out)
{
    uint64_t *d_a = nullptr, *d_b = nullptr;
    int *d_out = nullptr;
    cudaMalloc(&d_a, n * sizeof(uint64_t));
    cudaMalloc(&d_b, n * sizeof(uint64_t));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMemcpy(d_a, h_a, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    dim3 blk(256), grd((n + blk.x - 1) / blk.x);
    k_hamming<<<grd, blk>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

// --------------------------- 2) encoder -----------------------------------
__device__ __forceinline__ uint8_t nuc_lut(char c)
{
    // CPU mapping: A=0, C=1, G=2, T=3; everything else 0
    switch (c)
    {
    case 'C':
        return 1;
    case 'G':
        return 2;
    case 'T':
        return 3;
    default:
        return 0;
    }
}

__global__ void k_encode(const char *buf, int stride, int seqlen, uint64_t *sigs, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const char *s = buf + i * stride;
    uint64_t sig = 0;
#pragma unroll
    for (int j = 0; j < seqlen; ++j)
    {
        sig |= (uint64_t)nuc_lut(s[j]) << (j * 2);
    }
    sigs[i] = sig;
}

void gpu_encode_sequences(const char *h_buf, int n, int stride, int seqlen, uint64_t *h_out)
{
    char *d_buf = nullptr;
    uint64_t *d_out = nullptr;
    cudaMalloc(&d_buf, (size_t)n * stride);
    cudaMalloc(&d_out, n * sizeof(uint64_t));
    cudaMemcpy(d_buf, h_buf, (size_t)n * stride, cudaMemcpyHostToDevice);
    dim3 blk(256), grd((n + blk.x - 1) / blk.x);
    k_encode<<<grd, blk>>>(d_buf, stride, seqlen, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_buf);
    cudaFree(d_out);
}

// ---------------------- 3) distance scan (flat metadata) -------------------
__global__ void k_distance_scan(
    const uint64_t *__restrict__ d_querySigs, int queryCount,
    const uint64_t *__restrict__ d_offtargets,
    const uint64_t *__restrict__ d_allSignatures,   // packed [occ:32|id:32]
    const size_t *__restrict__ d_allSlicelistSizes, // concatenated per-slice
    const int *__restrict__ d_sliceLen,             // len per slice
    const size_t *__restrict__ d_sliceSizesOffset,  // per slice
    const size_t *__restrict__ d_sliceBaseOffset,   // per slice
    const uint32_t *__restrict__ d_prefixFlat,      // concatenated
    const size_t *__restrict__ d_prefixOffset,      // per slice
    const uint64_t *__restrict__ d_posIdxFlat,      // concatenated
    const size_t *__restrict__ d_posOffset,         // per slice
    int sliceCount, int maxDist,
    int *d_hitCount, int maxHits,
    int4 *d_hits, uint64_t *d_mismatches)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= queryCount)
        return;

    const uint64_t qsig = d_querySigs[q];

    for (int s = 0; s < sliceCount; ++s)
    {
        const int L = d_sliceLen[s];

        // build sub-code for this slice using its positions
        const uint64_t *pos = d_posIdxFlat + d_posOffset[s];
        uint64_t sub = 0ULL;
#pragma unroll
        for (int j = 0; j < 32; ++j)
        {
            if (j >= L)
                break;
            const uint64_t p = pos[j];
            sub |= ((qsig >> (p * 2)) & 3ULL) << (j * 2);
        }

        // locate sub-list (count + prefix begin) for (s, sub)
        const size_t sizesBase = d_sliceSizesOffset[s];
        const size_t count = d_allSlicelistSizes[sizesBase + sub];
        const size_t prefixBase = d_prefixOffset[s];
        const size_t begin = d_sliceBaseOffset[s] + (size_t)d_prefixFlat[prefixBase + sub];

        const uint64_t *ptr = d_allSignatures + begin;
        for (size_t t = 0; t < count; ++t)
        {
            const uint64_t packed = ptr[t];
            const uint32_t id = (uint32_t)(packed & 0xFFFFFFFFULL);
            const uint32_t occ = (uint32_t)(packed >> 32);

            const uint64_t x = qsig ^ d_offtargets[id];
            const uint64_t mism = ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1) | (x & 0x5555555555555555ULL);
            const int dist = __popcll(mism);

            if (dist <= maxDist)
            {
                const int out = atomicAdd(d_hitCount, 1);
                if (out < maxHits)
                {
                    d_hits[out] = make_int4(q, (int)id, (int)occ, (int)(mism >> 32)); // pack hi for quick copy
                    d_mismatches[out] = mism;
                }
            }
        }
    }
}

void gpu_distance_scan_flat(
    const std::vector<uint64_t> &querySigs,
    const std::vector<uint64_t> &offtargets,
    const std::vector<uint64_t> &allSignatures,
    const std::vector<size_t> &allSlicelistSizes,
    const std::vector<int> &sliceLen,
    const std::vector<size_t> &sliceSizesOffset,
    const std::vector<size_t> &sliceBaseOffset,
    const std::vector<uint32_t> &prefixFlat,
    const std::vector<size_t> &prefixOffset,
    const std::vector<uint64_t> &posIdxFlat,
    const std::vector<size_t> &posOffset,
    int maxDist,
    std::vector<Hit> &out_hits)
{
    const int Q = (int)querySigs.size();
    const int S = (int)sliceLen.size();

    uint64_t *d_off = nullptr, *dSig = nullptr, *dPos = nullptr;
    size_t *dSizes = nullptr, *dSizesOff = nullptr, *dBaseOff = nullptr, *dPrefOff = nullptr, *dPosOff = nullptr;
    int *dLen = nullptr;
    uint32_t *dPref = nullptr;

    cudaMalloc(&d_off, offtargets.size() * sizeof(uint64_t));
    cudaMalloc(&dSig, allSignatures.size() * sizeof(uint64_t));
    cudaMalloc(&dSizes, allSlicelistSizes.size() * sizeof(size_t));
    cudaMalloc(&dLen, sliceLen.size() * sizeof(int));
    cudaMalloc(&dSizesOff, sliceSizesOffset.size() * sizeof(size_t));
    cudaMalloc(&dBaseOff, sliceBaseOffset.size() * sizeof(size_t));
    cudaMalloc(&dPref, prefixFlat.size() * sizeof(uint32_t));
    cudaMalloc(&dPrefOff, prefixOffset.size() * sizeof(size_t));
    cudaMalloc(&dPos, posIdxFlat.size() * sizeof(uint64_t));
    cudaMalloc(&dPosOff, posOffset.size() * sizeof(size_t));

    cudaMemcpy(d_off, offtargets.data(), offtargets.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dSig, allSignatures.data(), allSignatures.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dSizes, allSlicelistSizes.data(), allSlicelistSizes.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dLen, sliceLen.data(), sliceLen.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dSizesOff, sliceSizesOffset.data(), sliceSizesOffset.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dBaseOff, sliceBaseOffset.data(), sliceBaseOffset.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dPref, prefixFlat.data(), prefixFlat.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dPrefOff, prefixOffset.data(), prefixOffset.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dPos, posIdxFlat.data(), posIdxFlat.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dPosOff, posOffset.data(), posOffset.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    const int BATCH = 2000; // tune to your GPU
    out_hits.clear();
    out_hits.reserve(Q * 64);

    for (int base = 0; base < Q; base += BATCH)
    {
        const int n = std::min(BATCH, Q - base);

        uint64_t *d_q = nullptr;
        cudaMalloc(&d_q, n * sizeof(uint64_t));
        cudaMemcpy(d_q, querySigs.data() + base, n * sizeof(uint64_t), cudaMemcpyHostToDevice);

        int *d_hitCount = nullptr;
        int4 *d_hits = nullptr;
        uint64_t *d_mism = nullptr;

        int capacity = std::max(1, n * 128);
        while (true)
        {
            if (!d_hitCount)
                cudaMalloc(&d_hitCount, sizeof(int));
            cudaMemset(d_hitCount, 0, sizeof(int));
            if (d_hits)
                cudaFree(d_hits);
            if (d_mism)
                cudaFree(d_mism);
            cudaMalloc(&d_hits, capacity * sizeof(int4));
            cudaMalloc(&d_mism, capacity * sizeof(uint64_t));

            dim3 blk(256), grd((n + blk.x - 1) / blk.x);
            k_distance_scan<<<grd, blk>>>(
                d_q, n, d_off, dSig, dSizes, dLen, dSizesOff, dBaseOff,
                dPref, dPrefOff, dPos, dPosOff,
                S, maxDist, d_hitCount, capacity, d_hits, d_mism);
            cudaDeviceSynchronize();

            int hCount = 0;
            cudaMemcpy(&hCount, d_hitCount, sizeof(int), cudaMemcpyDeviceToHost);

            if (hCount < capacity)
            {
                std::vector<int4> tmp(hCount);
                std::vector<uint64_t> mm(hCount);
                cudaMemcpy(tmp.data(), d_hits, hCount * sizeof(int4), cudaMemcpyDeviceToHost);
                cudaMemcpy(mm.data(), d_mism, hCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

                const size_t old = out_hits.size();
                out_hits.resize(old + hCount);
                for (int i = 0; i < hCount; ++i)
                {
                    out_hits[old + i] = {tmp[i].x + base, (uint32_t)tmp[i].y, (uint32_t)tmp[i].z, mm[i]};
                }
                break;
            }
            else
            {
                capacity = capacity * 2;
            }
        }

        cudaFree(d_q);
        cudaFree(d_hitCount);
        cudaFree(d_hits);
        cudaFree(d_mism);
    }

    cudaFree(d_off);
    cudaFree(dSig);
    cudaFree(dSizes);
    cudaFree(dLen);
    cudaFree(dSizesOff);
    cudaFree(dBaseOff);
    cudaFree(dPref);
    cudaFree(dPrefOff);
    cudaFree(dPos);
    cudaFree(dPosOff);
}

// ======================= 4) Dedup (q,id) and qOffset ========================
DedupResult gpu_dedup_by_qid(const std::vector<Hit>& hits, int Q)
{
    using thrust::device_vector;
    using thrust::make_tuple;
    using thrust::make_zip_iterator;

    DedupResult res;
    if (hits.empty()) {
        res.qOffset.assign(Q+1, 0);
        res.distinctCount = 0;
        return res;
    }

    const size_t H = hits.size();
    device_vector<int>      d_q(H);
    device_vector<uint32_t> d_id(H), d_occ(H);
    device_vector<uint64_t> d_mism(H);

    // copy in (host -> device)
    for (size_t i=0; i<H; ++i) {
        d_q[i]    = hits[i].q;
        d_id[i]   = hits[i].id;
        d_occ[i]  = hits[i].occ;
        d_mism[i] = hits[i].mismatches;
    }

    // sort by (q,id)
    auto keys_begin = make_zip_iterator(make_tuple(d_q.begin(), d_id.begin()));
    auto keys_end   = make_zip_iterator(make_tuple(d_q.end(),   d_id.end()));
    auto vals_begin = make_zip_iterator(make_tuple(d_occ.begin(), d_mism.begin()));
    thrust::sort_by_key(keys_begin, keys_end, vals_begin);

    // reduce_by_key to dedup (q,id), keep first occ+mism (no sum)
    device_vector<int>      q_u(d_q.size());
    device_vector<uint32_t> id_u(d_id.size());
    device_vector<uint32_t> occ_u(d_occ.size());
    device_vector<uint64_t> mism_u(d_mism.size());

    auto out_keys_begin = make_zip_iterator(make_tuple(q_u.begin(), id_u.begin()));
    auto out_vals_begin = make_zip_iterator(make_tuple(occ_u.begin(), mism_u.begin()));

    auto new_end = thrust::reduce_by_key(
        keys_begin, keys_end, vals_begin,
        out_keys_begin, out_vals_begin,
        KeyEqQid{}, OccPlusKeepMism{}
    );

    size_t U = new_end.first - out_keys_begin;
    q_u.resize(U); id_u.resize(U); occ_u.resize(U); mism_u.resize(U);

    // ----- Build qOffset correctly -----
    // 1) count hits per unique q (keep the q values)
    device_vector<int> q_unique(U);
    device_vector<int> q_counts_compact(U);
    auto end_counts = thrust::reduce_by_key(
        q_u.begin(), q_u.end(),
        thrust::make_constant_iterator<int>(1),
        q_unique.begin(),
        q_counts_compact.begin()
    );
    size_t K = end_counts.first - q_unique.begin();
    q_unique.resize(K);
    q_counts_compact.resize(K);

    // 2) scatter compact counts into dense histogram [0..Q-1]
    device_vector<int> q_counts(Q, 0);
    thrust::scatter(q_counts_compact.begin(), q_counts_compact.end(),
                    q_unique.begin(), q_counts.begin());

    // 3) exclusive scan to offsets of size Q+1
    device_vector<size_t> qOffset(Q+1);
    qOffset[0] = 0;
    thrust::exclusive_scan(q_counts.begin(), q_counts.end(), qOffset.begin()+1);

    // ----- Distinct IDs across all queries -----
    device_vector<uint32_t> id_copy = id_u;
    thrust::sort(id_copy.begin(), id_copy.end());
    auto id_end = thrust::unique(id_copy.begin(), id_copy.end());
    uint64_t distinct = static_cast<uint64_t>(id_end - id_copy.begin());

    // copy out to host
    res.q_u.resize(U);         thrust::copy(q_u.begin(),    q_u.end(),    res.q_u.begin());
    res.id_u.resize(U);        thrust::copy(id_u.begin(),   id_u.end(),   res.id_u.begin());
    res.occ_u.resize(U);       thrust::copy(occ_u.begin(),  occ_u.end(),  res.occ_u.begin());
    res.mism_u.resize(U);      thrust::copy(mism_u.begin(), mism_u.end(), res.mism_u.begin());
    res.qOffset.resize(Q+1);   thrust::copy(qOffset.begin(), qOffset.end(), res.qOffset.begin());
    res.distinctCount = distinct;
    return res;
}

// ======================= 5) Scoring kernel & wrappers =======================
__constant__ double d_cfdPam[16];
__constant__ double d_cfdPos[/* 20*16 or more; we will load exact size at runtime */ 1024]; // large enough guard

void gpu_load_cfd_tables(const double *pam, size_t pamSize, const double *pos, size_t posSize)
{
    if (pamSize > 16)
        throw std::runtime_error("cfdPamPenalties size > 16 unexpected");
    cudaMemcpyToSymbol(d_cfdPam, pam, pamSize * sizeof(double), 0, cudaMemcpyHostToDevice);
    // d_cfdPos array declared with upper bound; copy only posSize
    cudaMemcpyToSymbol(d_cfdPos, pos, posSize * sizeof(double), 0, cudaMemcpyHostToDevice);
}

__device__ __forceinline__ int popc64(uint64_t x) { return __popcll(x); }

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
                    if (((s)) != o)
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

    cudaMalloc(&d_query, Q * sizeof(uint64_t));
    cudaMalloc(&d_off, N * sizeof(uint64_t));
    cudaMalloc(&d_q_u, U * sizeof(int));
    cudaMalloc(&d_id_u, U * sizeof(uint32_t));
    cudaMalloc(&d_occ_u, U * sizeof(uint32_t));
    cudaMalloc(&d_mism_u, U * sizeof(uint64_t));
    cudaMalloc(&d_qOffset, (Q + 1) * sizeof(size_t));
    if (M > 0)
    {
        cudaMalloc(&d_mitMasks, M * sizeof(uint64_t));
        cudaMalloc(&d_mitVals, M * sizeof(double));
    }
    cudaMalloc(&d_outMIT, Q * sizeof(double));
    cudaMalloc(&d_outCFD, Q * sizeof(double));
    cudaMalloc(&d_outCount, Q * sizeof(uint64_t));

    // copy
    cudaMemcpy(d_query, querySigs.data(), Q * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, offtargets.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_u, q_u.data(), U * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_id_u, id_u.data(), U * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_occ_u, occ_u.data(), U * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mism_u, mism_u.data(), U * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qOffset, qOffset.data(), (Q + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    if (M > 0)
    {
        cudaMemcpy(d_mitMasks, mitMasks.data(), M * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mitVals, mitVals.data(), M * sizeof(double), cudaMemcpyHostToDevice);
    }

    // launch (one block per query)
    dim3 grd(Q), blk(128);
    k_score_queries<<<grd, blk>>>(
        d_query, d_off, d_q_u, d_id_u, d_occ_u, d_mism_u, d_qOffset,
        d_mitMasks, d_mitVals, M, threshold, scoreMethod,
        calcMit ? 1 : 0, calcCfd ? 1 : 0,
        d_outMIT, d_outCFD, d_outCount);
    cudaDeviceSynchronize();

    // copy back
    outMIT.resize(Q);
    outCFD.resize(Q);
    outCount.resize(Q);
    cudaMemcpy(outMIT.data(), d_outMIT, Q * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(outCFD.data(), d_outCFD, Q * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(outCount.data(), d_outCount, Q * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_query);
    cudaFree(d_off);
    cudaFree(d_q_u);
    cudaFree(d_id_u);
    cudaFree(d_occ_u);
    cudaFree(d_mism_u);
    cudaFree(d_qOffset);
    if (M > 0)
    {
        cudaFree(d_mitMasks);
        cudaFree(d_mitVals);
    }
    cudaFree(d_outMIT);
    cudaFree(d_outCFD);
    cudaFree(d_outCount);
}
