#pragma once
#include <cstdint>
#include <vector>

// 1) Popcount smoke test
void gpu_popcount_hamming(const uint64_t* h_a, const uint64_t* h_b, int n, int* h_out);

// 2) GPU sequence -> signature (20-mer) encoder; copies back to host buffer
void gpu_encode_sequences(const char* h_buf, int n, int seqLineLength, uint64_t* h_out);

// 3) Hit structure returned by GPU distance stage
struct Hit { int q; uint32_t id; uint32_t occ; uint64_t mismatches; };

// 4) GPU distance-only scan using *flat* metadata (no host pointers)
//    Produces compact hits (q, id, occ, mismatches) with dist <= maxDist.
void gpu_distance_scan_flat(
    const std::vector<uint64_t>& querySigs,
    const std::vector<uint64_t>& offtargets,
    const std::vector<uint64_t>& allSignatures,      // packed [occ:32|id:32]
    const std::vector<size_t>&   allSlicelistSizes,  // concatenated per-slice sublist sizes
    const std::vector<int>&      sliceLen,           // per slice
    const std::vector<size_t>&   sliceSizesOffset,   // per slice: index into allSlicelistSizes
    const std::vector<size_t>&   sliceBaseOffset,    // per slice: base offset into allSignatures
    const std::vector<uint32_t>& prefixFlat,         // concatenated prefix sums per slice
    const std::vector<size_t>&   prefixOffset,       // per slice: start index into prefixFlat
    const std::vector<uint64_t>& posIdxFlat,         // concatenated positions (from sliceMasks)
    const std::vector<size_t>&   posOffset,          // per slice: start index into posIdxFlat
    int maxDist,
    std::vector<Hit>& out_hits);
