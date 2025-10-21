// ISSLScoreOfftargets/issl_cuda.cuh
#pragma once
#include <cstdint>
#include <vector>
#include <string>

// 1) Popcount smoke test
void gpu_popcount_hamming(const uint64_t *h_a, const uint64_t *h_b, int n, int *h_out);

// 2) GPU sequence -> signature encoder (uses seqLen from index; exact CPU parity)
void gpu_encode_sequences(const char *h_buf, int n, int seqLineLength, int seqLen, uint64_t *h_out);

// 3) Hit structure returned by GPU distance stage
struct Hit
{
    int q;
    uint32_t id;
    uint32_t occ;
    uint64_t mismatches;
};

// 4) GPU distance-only scan using *flat* metadata (no host pointers)
//    Produces compact hits (q, id, occ, mismatches) with dist <= maxDist.
void gpu_distance_scan_flat(
    const std::vector<uint64_t> &querySigs,
    const std::vector<uint64_t> &offtargets,
    const std::vector<uint64_t> &allSignatures,   // packed [occ:32|id:32]
    const std::vector<size_t> &allSlicelistSizes, // concatenated per-slice sublist sizes
    const std::vector<int> &sliceLen,             // per slice
    const std::vector<size_t> &sliceSizesOffset,  // per slice: index into allSlicelistSizes
    const std::vector<size_t> &sliceBaseOffset,   // per slice: base offset into allSignatures
    const std::vector<uint32_t> &prefixFlat,      // concatenated prefix sums per slice
    const std::vector<size_t> &prefixOffset,      // per slice: start index into prefixFlat
    const std::vector<uint64_t> &posIdxFlat,      // concatenated positions (from sliceMasks)
    const std::vector<size_t> &posOffset,         // per slice: start index into posIdxFlat
    int maxDist,
    std::vector<Hit> &out_hits);

// ==================== NEW: GPU dedup + scoring API ==========================
struct DedupResult {
    std::vector<int>       q_u;      // per-hit query index after dedup
    std::vector<uint32_t>  id_u;     // unique id
    std::vector<uint32_t>  occ_u;    // summed occurrences
    std::vector<uint64_t>  mism_u;   // representative mismatch mask (same as CPU after dedup)
    std::vector<size_t>    qOffset;  // size Q+1
    uint64_t               distinctCount; // unique IDs across all queries
};

// Deduplicate (q,id), sum occ, keep first mism; also build qOffset and distinct IDs.
DedupResult gpu_dedup_by_qid(const std::vector<Hit>& hits, int Q);

// Upload CFD tables to __constant__ memory
void gpu_load_cfd_tables(const double* pam, size_t pamSize, const double* pos, size_t posSize);

// GPU scoring: per-query block, early-exit parity, MIT via LUT
void gpu_score_queries(
    const std::vector<uint64_t>& querySigs,
    const std::vector<uint64_t>& offtargets,
    const std::vector<int>&      q_u,
    const std::vector<uint32_t>& id_u,
    const std::vector<uint32_t>& occ_u,
    const std::vector<uint64_t>& mism_u,
    const std::vector<size_t>&   qOffset,          // Q+1
    const std::vector<uint64_t>& mitMasks,         // sorted (host LUT masks)
    const std::vector<double>&   mitVals,          // same length as mitMasks
    double threshold, int scoreMethod,
    bool calcMit, bool calcCfd,
    std::vector<double>& outMIT, std::vector<double>& outCFD, std::vector<uint64_t>& outCount);

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
    std::vector<Hit>& out_hits);

    