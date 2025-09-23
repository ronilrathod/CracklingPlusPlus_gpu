#pragma once
#include <cstdint>
#include <vector>

// 1) Popcount smoke test
void gpu_popcount_hamming(const uint64_t* h_a, const uint64_t* h_b, int n, int* h_out);

// 2) GPU sequence -> signature (20-mer) encoder; copies back to host buffer
void gpu_encode_sequences(const char* h_buf, int n, int seqLineLength, uint64_t* h_out);

// 3) Baseline GPU scoring inner loop (distance only, dist<=maxDist)
//    Returns a compact vector of candidate hits to be scored on CPU:
//       (queryIdx, signatureId, occurrences, mismatchesMask)
struct Hit { int q; uint32_t id; uint32_t occ; uint64_t mismatches; };
void gpu_scan_slices_distance_only(
    const uint64_t* h_querySignatures, int queryCount,
    const uint64_t* h_offtargets, int offtargetsCount,
    const size_t*    h_allSlicelistSizes,  // length = sum over slices of 4^(sliceLen)
    uint64_t* const* h_sliceListPtrs,      // per-slice array of pointers into allSignatures
    const int*       h_sliceLens,          // per-slice length in positions (e.g., 5,6,...)
    const uint64_t*  h_slicePosIdx,        // flattened [sum over slices of sliceLen] positions
    int sliceCount,
    int maxDist,
    std::vector<Hit>& out_hits);           // filled on host
