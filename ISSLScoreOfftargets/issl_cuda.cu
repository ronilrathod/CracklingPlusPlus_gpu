// ISSLScoreOfftargets/issl_cuda.cu

#include "issl_cuda.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// --------------------------- 1) popcount sanity ----------------------------
__global__ void k_hamming(const uint64_t* a, const uint64_t* b, int* out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __popcll(a[i] ^ b[i]);
}

void gpu_popcount_hamming(const uint64_t* h_a, const uint64_t* h_b, int n, int* h_out){
    uint64_t *d_a=nullptr,*d_b=nullptr; int* d_out=nullptr;
    cudaMalloc(&d_a, n*sizeof(uint64_t));
    cudaMalloc(&d_b, n*sizeof(uint64_t));
    cudaMalloc(&d_out, n*sizeof(int));
    cudaMemcpy(d_a, h_a, n*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*sizeof(uint64_t), cudaMemcpyHostToDevice);
    dim3 blk(256), grd((n+blk.x-1)/blk.x);
    k_hamming<<<grd,blk>>>(d_a,d_b,d_out,n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

// --------------------------- 2) encoder -----------------------------------
__device__ __forceinline__ uint8_t nuc_lut(char c){
    // CPU mapping: A=0, C=1, G=2, T=3; everything else 0
    switch(c){ case 'C': return 1; case 'G': return 2; case 'T': return 3; default: return 0; }
}

__global__ void k_encode(const char* buf, int stride, int seqlen, uint64_t* sigs, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const char* s = buf + i*stride;
    uint64_t sig = 0;
    #pragma unroll
    for (int j=0; j<seqlen; ++j){
        sig |= (uint64_t)nuc_lut(s[j]) << (j*2);
    }
    sigs[i] = sig;
}

void gpu_encode_sequences(const char* h_buf, int n, int stride, int seqlen, uint64_t* h_out){
    char* d_buf=nullptr; uint64_t* d_out=nullptr;
    cudaMalloc(&d_buf, (size_t)n*stride);
    cudaMalloc(&d_out, n*sizeof(uint64_t));
    cudaMemcpy(d_buf, h_buf, (size_t)n*stride, cudaMemcpyHostToDevice);
    dim3 blk(256), grd((n+blk.x-1)/blk.x);
    k_encode<<<grd,blk>>>(d_buf, stride, seqlen, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, n*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_buf); cudaFree(d_out);
}

// ---------------------- 3) distance scan (flat metadata) -------------------
__global__ void k_distance_scan(
    const uint64_t* __restrict__ d_querySigs, int queryCount,
    const uint64_t* __restrict__ d_offtargets,
    const uint64_t* __restrict__ d_allSignatures,      // packed [occ:32|id:32]
    const size_t*   __restrict__ d_allSlicelistSizes,  // concatenated per-slice
    const int*      __restrict__ d_sliceLen,           // len per slice
    const size_t*   __restrict__ d_sliceSizesOffset,   // per slice
    const size_t*   __restrict__ d_sliceBaseOffset,    // per slice
    const uint32_t* __restrict__ d_prefixFlat,         // concatenated
    const size_t*   __restrict__ d_prefixOffset,       // per slice
    const uint64_t* __restrict__ d_posIdxFlat,         // concatenated
    const size_t*   __restrict__ d_posOffset,          // per slice
    int sliceCount, int maxDist,
    int* d_hitCount, int maxHits,
    int4* d_hits, uint64_t* d_mismatches)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= queryCount) return;

    const uint64_t qsig = d_querySigs[q];

    for (int s = 0; s < sliceCount; ++s) {
        const int L = d_sliceLen[s];

        // build sub-code for this slice using its positions
        const uint64_t* pos = d_posIdxFlat + d_posOffset[s];
        uint64_t sub = 0ULL;
        #pragma unroll
        for (int j=0; j<32; ++j){
            if (j>=L) break;
            const uint64_t p = pos[j];
            sub |= ((qsig >> (p*2)) & 3ULL) << (j*2);
        }

        // locate sub-list (count + prefix begin) for (s, sub)
        const size_t sizesBase  = d_sliceSizesOffset[s];
        const size_t count      = d_allSlicelistSizes[sizesBase + sub];
        const size_t prefixBase = d_prefixOffset[s];
        const size_t begin      = d_sliceBaseOffset[s] + (size_t)d_prefixFlat[prefixBase + sub];

        const uint64_t* ptr = d_allSignatures + begin;
        for (size_t t = 0; t < count; ++t) {
            const uint64_t packed = ptr[t];
            const uint32_t id  = (uint32_t)(packed & 0xFFFFFFFFULL);
            const uint32_t occ = (uint32_t)(packed >> 32);

            const uint64_t x    = qsig ^ d_offtargets[id];
            const uint64_t mism = ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1) | (x & 0x5555555555555555ULL);
            const int dist      = __popcll(mism);

            if (dist <= maxDist) {
                const int out = atomicAdd(d_hitCount, 1);
                if (out < maxHits) {
                    d_hits[out]       = make_int4(q, (int)id, (int)occ, (int)(mism >> 32));
                    d_mismatches[out] = mism;
                }
            }
        }
    }
}

void gpu_distance_scan_flat(
    const std::vector<uint64_t>& querySigs,
    const std::vector<uint64_t>& offtargets,
    const std::vector<uint64_t>& allSignatures,
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
    // ---- Static metadata to device once ----
    const int Q = (int)querySigs.size();
    const int S = (int)sliceLen.size();

    uint64_t *d_off=nullptr,*dSig=nullptr,*dPos=nullptr;
    size_t   *dSizes=nullptr,*dSizesOff=nullptr,*dBaseOff=nullptr,*dPrefOff=nullptr,*dPosOff=nullptr;
    int      *dLen=nullptr;
    uint32_t *dPref=nullptr;

    cudaMalloc(&d_off,    offtargets.size()*sizeof(uint64_t));
    cudaMalloc(&dSig,     allSignatures.size()*sizeof(uint64_t));
    cudaMalloc(&dSizes,   allSlicelistSizes.size()*sizeof(size_t));
    cudaMalloc(&dLen,     sliceLen.size()*sizeof(int));
    cudaMalloc(&dSizesOff,  sliceSizesOffset.size()*sizeof(size_t));
    cudaMalloc(&dBaseOff,   sliceBaseOffset.size()*sizeof(size_t));
    cudaMalloc(&dPref,      prefixFlat.size()*sizeof(uint32_t));
    cudaMalloc(&dPrefOff,   prefixOffset.size()*sizeof(size_t));
    cudaMalloc(&dPos,       posIdxFlat.size()*sizeof(uint64_t));
    cudaMalloc(&dPosOff,    posOffset.size()*sizeof(size_t));

    cudaMemcpy(d_off,    offtargets.data(),    offtargets.size()*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dSig,     allSignatures.data(), allSignatures.size()*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dSizes,   allSlicelistSizes.data(), allSlicelistSizes.size()*sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dLen,     sliceLen.data(),      sliceLen.size()*sizeof(int),         cudaMemcpyHostToDevice);
    cudaMemcpy(dSizesOff, sliceSizesOffset.data(), sliceSizesOffset.size()*sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dBaseOff,  sliceBaseOffset.data(),  sliceBaseOffset.size()*sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dPref,     prefixFlat.data(),       prefixFlat.size()*sizeof(uint32_t),    cudaMemcpyHostToDevice);
    cudaMemcpy(dPrefOff,  prefixOffset.data(),     prefixOffset.size()*sizeof(size_t),    cudaMemcpyHostToDevice);
    cudaMemcpy(dPos,      posIdxFlat.data(),       posIdxFlat.size()*sizeof(uint64_t),    cudaMemcpyHostToDevice);
    cudaMemcpy(dPosOff,   posOffset.data(),        posOffset.size()*sizeof(size_t),       cudaMemcpyHostToDevice);

    // ---- Batch queries to bound memory & avoid truncation ----
    const int BATCH = 2000; // tune to your GPU
    out_hits.clear();
    out_hits.reserve(Q * 64); // guess; will grow as needed

    for (int base = 0; base < Q; base += BATCH) {
        const int n = std::min(BATCH, Q - base);

        // device queries for this batch
        uint64_t *d_q=nullptr;
        cudaMalloc(&d_q, n*sizeof(uint64_t));
        cudaMemcpy(d_q, querySigs.data()+base, n*sizeof(uint64_t), cudaMemcpyHostToDevice);

        // auto-grow output buffers if we overflow
        int*      d_hitCount=nullptr;
        int4*     d_hits=nullptr;
        uint64_t* d_mism=nullptr;

        int capacity = std::max(1, n * 128); // start generous per batch
        while (true) {
            // (re)allocate
            if (!d_hitCount) cudaMalloc(&d_hitCount, sizeof(int));
            cudaMemset(d_hitCount, 0, sizeof(int));
            if (d_hits) cudaFree(d_hits);
            if (d_mism) cudaFree(d_mism);
            cudaMalloc(&d_hits, capacity * sizeof(int4));
            cudaMalloc(&d_mism, capacity * sizeof(uint64_t));

            // launch
            dim3 blk(256), grd((n + blk.x - 1) / blk.x);
            k_distance_scan<<<grd, blk>>>(
                d_q, n, d_off, dSig, dSizes, dLen, dSizesOff, dBaseOff,
                dPref, dPrefOff, dPos, dPosOff,
                S, maxDist, d_hitCount, capacity, d_hits, d_mism);
            cudaDeviceSynchronize();

            int hCount=0; cudaMemcpy(&hCount, d_hitCount, sizeof(int), cudaMemcpyDeviceToHost);

            if (hCount < capacity) {
                // success: copy back this batch
                std::vector<int4>    tmp(hCount);
                std::vector<uint64_t> mm(hCount);
                cudaMemcpy(tmp.data(),  d_hits, hCount*sizeof(int4),     cudaMemcpyDeviceToHost);
                cudaMemcpy(mm.data(),   d_mism, hCount*sizeof(uint64_t),  cudaMemcpyDeviceToHost);

                // append to out_hits; fix q by adding base
                const size_t old = out_hits.size();
                out_hits.resize(old + hCount);
                for (int i=0; i<hCount; ++i) {
                    out_hits[old + i] = { tmp[i].x + base, (uint32_t)tmp[i].y, (uint32_t)tmp[i].z, mm[i] };
                }
                break; // batch done
            } else {
                // overflow: grow and retry
                capacity = capacity * 2;
            }
        }

        cudaFree(d_q);
        cudaFree(d_hitCount);
        cudaFree(d_hits);
        cudaFree(d_mism);
    }

    // cleanup static metadata
    cudaFree(d_off); cudaFree(dSig); cudaFree(dSizes);
    cudaFree(dLen);  cudaFree(dSizesOff); cudaFree(dBaseOff);
    cudaFree(dPref); cudaFree(dPrefOff); cudaFree(dPos); cudaFree(dPosOff);
}
