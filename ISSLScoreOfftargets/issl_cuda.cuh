// ISSLScoreOfftargets/issl_cuda.cuh
// GPU CUDA header for ISSL off-target scoring
#pragma once
#include <cstdint>
#include <vector>
#include <string>

// ==================== DATA STRUCTURES ====================

// Hit structure returned by GPU distance stage
struct Hit
{
    int q;              // query index
    uint32_t id;        // off-target ID
    uint32_t occ;       // occurrence count
    uint64_t mismatches; // mismatch pattern
};

// Result structure from deduplication process
struct DedupResult {
    std::vector<int>       q_u;      // per-hit query index after dedup
    std::vector<uint32_t>  id_u;     // unique id
    std::vector<uint32_t>  occ_u;    // summed occurrences
    std::vector<uint64_t>  mism_u;   // representative mismatch mask (same as CPU after dedup)
    std::vector<size_t>    qOffset;  // size Q+1
    uint64_t               distinctCount; // unique IDs across all queries
};

// ==================== FUNCTION DECLARATIONS ====================

// GPU sequence encoding: convert DNA sequences to 64-bit signatures
void gpu_encode_sequences(const char *h_buf, int n, int seqLineLength, int seqLen, uint64_t *h_out);

// GPU distance scanning: find off-targets within maximum distance
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

// GPU deduplication: remove duplicate (query, off-target) pairs
DedupResult gpu_dedup_by_qid(const std::vector<Hit>& hits, int Q);

// Load CFD penalty tables into GPU constant memory
void gpu_load_cfd_tables(const double* pam, size_t pamSize, const double* pos, size_t posSize);

// GPU scoring: calculate MIT and CFD scores with early exit optimization
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