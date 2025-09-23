#include "issl_cuda.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cassert>

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
    // Matches your CPU mapping: A=0, C=1, G=2, T=3; everything else 0
    switch(c){ case 'C': return 1; case 'G': return 2; case 'T': return 3; default: return 0; }
}
__global__ void k_encode(const char* buf, int stride, uint64_t* sigs, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const char* s = buf + i*stride;
    uint64_t sig = 0;
    #pragma unroll
    for (int j=0; j<20; ++j){
        sig |= (uint64_t)nuc_lut(s[j]) << (j*2);
    }
    sigs[i] = sig;
}
void gpu_encode_sequences(const char* h_buf, int n, int stride, uint64_t* h_out){
    char* d_buf=nullptr; uint64_t* d_out=nullptr;
    cudaMalloc(&d_buf, (size_t)n*stride);
    cudaMalloc(&d_out, n*sizeof(uint64_t));
    cudaMemcpy(d_buf, h_buf, (size_t)n*stride, cudaMemcpyHostToDevice);
    dim3 blk(256), grd((n+blk.x-1)/blk.x);
    k_encode<<<grd,blk>>>(d_buf, stride, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, n*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_buf); cudaFree(d_out);
}

// --------------------------- 3) baseline GPU inner loop --------------------
// We pass a flattened view of per-slice metadata to avoid complex host->device graphs.
// Device copies of per-slice info:
struct SliceMeta {
    const uint64_t* sliceBase; // start of this slice's signatures (uint64_t packed: [occ:32|id:32])
    int sliceLen;              // number of positions in this slice (e.g., 5,6,...)
    size_t listBaseIdx;        // starting index into allSlicelistSizes for this slice
};
__device__ __forceinline__ uint64_t build_search_slice(uint64_t searchSig, const uint64_t* posIdx, int sliceLen){
    // posIdx points to the positions (e.g., [3,7,9,...]) for this slice
    uint64_t code = 0ULL;
    #pragma unroll
    for (int j=0; j<32; ++j){ // upper bound; early break
        if (j>=sliceLen) break;
        uint64_t p = posIdx[j];
        code |= ((searchSig >> (p*2)) & 3ULL) << (j*2);
    }
    return code;
}

__global__ void k_scan_slices_distance_only(
    const uint64_t* querySigs, int queryCount,
    const uint64_t* offtargets, int offtargetsCount,
    const size_t* allSizes,  // concatenated sizes for all slices
    const SliceMeta* metas,  // length = sliceCount
    const uint64_t* allPosIdx, // concatenated positions for all slices
    const size_t* posOffsets,  // prefix offsets into allPosIdx per slice (length=sliceCount+1)
    int sliceCount, int maxDist,
    // output
    int* d_hitCount,
    int  maxHits,
    int4* d_hits,            // (q, id, occ, hi32(mismatches)) low32 of mismatches lost; store full separately:
    uint64_t* d_mismatch64)  // store mismatches mask per hit
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= queryCount) return;

    uint64_t qsig = querySigs[q];

    for (int s=0; s<sliceCount; ++s){
        const SliceMeta sm = metas[s];
        const uint64_t* posIdx = allPosIdx + posOffsets[s];

        uint64_t searchSlice = build_search_slice(qsig, posIdx, sm.sliceLen);
        size_t idx = sm.listBaseIdx + searchSlice;

        size_t count = allSizes[idx];
        const uint64_t* slicePtr = sm.sliceBase + (/*element offset*/0); // sm.sliceBase already points to start of this slice's signatures sequence
        // To index into this slice's block: sum(allSizes[sm.listBaseIdx .. sm.listBaseIdx + searchSlice - 1]).
        // For simplicity in this baseline, we assume sm.sliceBase already points to the first entry
        // for searchSlice. If not, precompute pointers on host and pass via metas[]. (Recommended.)

        const uint64_t* p = slicePtr; // pointer to first signature for this sub-list
        for (size_t j=0; j<count; ++j){
            uint64_t packed = p[j];
            uint32_t id  = (uint32_t)(packed & 0xFFFFFFFFULL);
            uint32_t occ = (uint32_t)(packed >> 32);

            // distance
            uint64_t x = qsig ^ offtargets[id];
            uint64_t mism = ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1) | (x & 0x5555555555555555ULL);
            int dist = __popcll(mism);
            if (dist <= maxDist){
                int myIdx = atomicAdd(d_hitCount, 1);
                if (myIdx < maxHits){
                    d_hits[myIdx] = make_int4(q, (int)id, (int)occ, (int)(mism >> 32));
                    d_mismatch64[myIdx] = mism;
                }
            }
        }
    }
}

void gpu_scan_slices_distance_only(
    const uint64_t* h_querySignatures, int queryCount,
    const uint64_t* h_offtargets, int offtargetsCount,
    const size_t*    h_allSlicelistSizes,
    uint64_t* const* h_sliceListPtrs,
    const int*       h_sliceLens,
    const uint64_t*  h_slicePosIdx,
    int sliceCount,
    int maxDist,
    std::vector<Hit>& out_hits)
{
    // --- Host->Device copies of big arrays
    uint64_t *d_q=nullptr, *d_off=nullptr;
    size_t   *d_sizes=nullptr;
    cudaMalloc(&d_q,   queryCount*sizeof(uint64_t));
    cudaMalloc(&d_off, offtargetsCount*sizeof(uint64_t));

    // Flatten slice metadata for device
    std::vector<SliceMeta> h_metas(sliceCount);
    // IMPORTANT: for this baseline we assume h_sliceListPtrs[s] ALREADY points to the first element for each specific sub-list
    // If not, precompute per-(slice,subcode) pointers on host and pass them (more memory, simpler kernel).
    // To keep it simple here, we assume host prepared metas[s].sliceBase such that adding j walks the chosen sub-list.

    // You already have per-slice vectors. For the baseline demo, we set sliceBase = h_sliceListPtrs[s].
    for (int s=0; s<sliceCount; ++s){
        h_metas[s].sliceBase  = h_sliceListPtrs[s];
        h_metas[s].sliceLen   = h_sliceLens[s];
        // listBaseIdx is the running starting index inside h_allSlicelistSizes for this slice (you already compute sliceLimitOffset on CPU)
        // Pass that exact offset here:
        // For this header-only sketch, set to 0; you'll wire the real offset from your CPU code.
        h_metas[s].listBaseIdx = 0;
    }

    // Copy flat arrays
    cudaMalloc(&d_sizes, /*size you computed on host*/ 0); // TODO: set correct bytes and copy
    cudaMemcpy(d_q, h_querySignatures, queryCount*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, h_offtargets,     offtargetsCount*sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Copy metas/positions
    SliceMeta* d_metas=nullptr; cudaMalloc(&d_metas, sliceCount*sizeof(SliceMeta));
    cudaMemcpy(d_metas, h_metas.data(), sliceCount*sizeof(SliceMeta), cudaMemcpyHostToDevice);

    // Positions arrays (flattened) + prefix offsets
    uint64_t* d_posIdx=nullptr; size_t* d_posOff=nullptr;
    // TODO: allocate/copy for your real sizes.

    // --- Output buffers (pre-allocate a generous cap; resize after)
    const int MAX_HITS = 1<<26; // adjust based on genome and workload; you can loop in batches too
    int* d_hitCount=nullptr; cudaMalloc(&d_hitCount, sizeof(int));
    cudaMemset(d_hitCount, 0, sizeof(int));
    int4* d_hits=nullptr; cudaMalloc(&d_hits, MAX_HITS*sizeof(int4));
    uint64_t* d_m64=nullptr; cudaMalloc(&d_m64, MAX_HITS*sizeof(uint64_t));

    // --- Launch
    dim3 blk(256), grd((queryCount+blk.x-1)/blk.x);
    k_scan_slices_distance_only<<<grd,blk>>>(
        d_q, queryCount, d_off, offtargetsCount,
        d_sizes, d_metas, d_posIdx, d_posOff, sliceCount, maxDist,
        d_hitCount, MAX_HITS, d_hits, d_m64);
    cudaDeviceSynchronize();

    // --- Copy back
    int h_hitCount=0; cudaMemcpy(&h_hitCount, d_hitCount, sizeof(int), cudaMemcpyDeviceToHost);
    h_hitCount = std::max(0, std::min(h_hitCount, MAX_HITS));
    std::vector<int4>   h_hits(h_hitCount);
    std::vector<uint64_t> h_mism(h_hitCount);
    cudaMemcpy(h_hits.data(), d_hits, h_hitCount*sizeof(int4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mism.data(), d_m64, h_hitCount*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    out_hits.clear(); out_hits.reserve(h_hitCount);
    for (int i=0;i<h_hitCount;++i){
        Hit h; h.q = h_hits[i].x; h.id = (uint32_t)h_hits[i].y; h.occ = (uint32_t)h_hits[i].z; h.mismatches = h_mism[i];
        out_hits.push_back(h);
    }

    // --- Cleanup
    cudaFree(d_q); cudaFree(d_off); cudaFree(d_sizes);
    cudaFree(d_metas); cudaFree(d_posIdx); cudaFree(d_posOff);
    cudaFree(d_hitCount); cudaFree(d_hits); cudaFree(d_m64);
}
