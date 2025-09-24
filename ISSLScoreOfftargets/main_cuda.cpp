// ISSLScoreOfftargets/main_cuda.cpp
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <climits> // for CHAR_BIT

#include "issl_cuda.cuh"    // Hit, gpu_encode_sequences(...), gpu_distance_scan_flat(...)
#include "../include/otScorePenalties.hpp" // same as CPU

// --- tiny helpers (no Boost) ------------------------------------------------
static inline std::string signatureToSequence(uint64_t sig, int seqLen)
{
    static const char LUT[4] = {'A', 'C', 'G', 'T'};
    std::string s;
    s.resize(seqLen);
    for (int j = 0; j < seqLen; ++j)
        s[j] = LUT[(sig >> (j * 2)) & 0x3];
    return s;
}

#if defined(_MSC_VER)
#include <intrin.h>
static inline int popcount64_host(uint64_t x) { return (int)__popcnt64(x); }
#else
static inline int popcount64_host(uint64_t x) { return __builtin_popcountll((unsigned long long)x); }
#endif

// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        std::fprintf(stderr,
                     "Usage: %s [issltable] [query file] [max distance] [score-threshold] [score-method] [output-file-optional]\n",
                     argv[0]);
        return 1;
    }

    const char *isslPath = argv[1];
    const char *queryPath = argv[2];
    const int maxDistCLI = std::atoi(argv[3]);
    const int maxDist = 4; // CPU uses dist <= 4
    if (maxDistCLI != 4)
    {
        std::fprintf(stderr, "[GPU] Note: clamping maxDist from %d to 4 to match CPU logic.\n", maxDistCLI);
    }
    const double threshold = std::atof(argv[4]); // used for early-exit in reduce
    const std::string scoreMethod = argv[5];
    const bool wantOutputName = (argc > 6);

    // Decide which scores to compute (mirror CPU flags)
    bool calcMit = false, calcCfd = false;
    if (scoreMethod == "and" || scoreMethod == "or" || scoreMethod == "avg")
    {
        calcMit = calcCfd = true;
    }
    else if (scoreMethod == "mit")
    {
        calcMit = true;
    }
    else if (scoreMethod == "cfd")
    {
        calcCfd = true;
    }
    else
    {
        std::fprintf(stderr, "Invalid scoring method. Acceptable: and|or|avg|mit|cfd\n");
        return 1;
    }

    // ---------------------------
    // 1) LOAD ISSL INDEX (CPU)
    // ---------------------------
    auto t_total_start = std::chrono::high_resolution_clock::now();
    auto t_load_start = t_total_start;

    FILE *isslFp = std::fopen(isslPath, "rb");
    if (!isslFp)
    {
        std::fprintf(stderr, "Error: could not open ISSL file: %s\n", isslPath);
        return 1;
    }

    // header: [offtargetsCount, seqLength, sliceCount]
    std::vector<size_t> slicelistHeader(3);
    if (std::fread(slicelistHeader.data(), sizeof(size_t), slicelistHeader.size(), isslFp) == 0)
    {
        std::fprintf(stderr, "Error: invalid ISSL header\n");
        return 1;
    }
    size_t offtargetsCount = slicelistHeader[0];
    size_t seqLength = slicelistHeader[1]; // e.g., 20
    size_t sliceCount = slicelistHeader[2];

    // offtargets (uint64_t per site)
    std::vector<uint64_t> offtargets(offtargetsCount);
    if (std::fread(offtargets.data(), sizeof(uint64_t), offtargetsCount, isslFp) == 0)
    {
        std::fprintf(stderr, "Error: loading off-target sequences failed\n");
        return 1;
    }

    // slice masks (one uint64 maskBinary per slice) -> expand to positions
    std::vector<std::vector<uint64_t>> sliceMasks;
    sliceMasks.reserve(sliceCount);
    for (size_t i = 0; i < sliceCount; ++i)
    {
        uint64_t maskBinary = 0;
        if (std::fread(&maskBinary, sizeof(uint64_t), 1, isslFp) != 1)
        {
            std::fprintf(stderr, "Error: reading slice mask %zu failed\n", i);
            return 1;
        }
        std::vector<uint64_t> mask;
        mask.reserve(seqLength);
        for (uint64_t j = 0; j < seqLength; ++j)
            if (maskBinary & (1ULL << j))
                mask.push_back(j);
        sliceMasks.push_back(std::move(mask));
    }

    // total lists across all slices
    size_t sliceListCount = 0;
    for (size_t i = 0; i < sliceCount; ++i)
        sliceListCount += (1ULL << (sliceMasks[i].size() * 2));

    // per-sublist sizes (concatenated over slices)
    std::vector<size_t> allSlicelistSizes(sliceListCount);

    // all signatures per slice stored contiguously.
    // Each entry is packed uint64_t: [occ:32 | id:32]
    std::vector<uint64_t> allSignatures(offtargetsCount * sliceCount);

    // read per-slice blocks
    size_t sizesCursor = 0;
    for (size_t i = 0; i < sliceCount; ++i)
    {
        const size_t subCount = 1ULL << (sliceMasks[i].size() * 2);
        if (std::fread(allSlicelistSizes.data() + sizesCursor, sizeof(size_t), subCount, isslFp) == 0)
        {
            std::fprintf(stderr, "Error: reading slice list sizes (slice %zu) failed\n", i);
            return 1;
        }
        if (std::fread(allSignatures.data() + (offtargetsCount * i), sizeof(uint64_t), offtargetsCount, isslFp) == 0)
        {
            std::fprintf(stderr, "Error: reading slice contents (slice %zu) failed\n", i);
            return 1;
        }
        sizesCursor += subCount;
    }
    std::fclose(isslFp);

    // ---------------------------------------------
    // 2) PACK FLAT ARRAYS for the GPU distance scan
    // ---------------------------------------------
    std::vector<int> h_sliceLen(sliceCount);
    std::vector<size_t> h_sliceSizesOffset(sliceCount); // index into allSlicelistSizes for this slice
    std::vector<size_t> h_sliceBaseOffset(sliceCount);  // base offset (in elements) into allSignatures for this slice
    std::vector<uint64_t> h_posIdx_flat;                // concat of all sliceMasks[i]
    std::vector<size_t> h_posOffset(sliceCount + 1, 0); // start index into h_posIdx_flat per slice
    std::vector<uint32_t> h_prefix_flat;                // per-slice prefix sums over sublists
    std::vector<size_t> h_prefixOffset(sliceCount + 1, 0);

    size_t sizesCur = 0, sigCur = 0, prefixCur = 0;
    h_posIdx_flat.reserve(sliceCount * 20);
    for (size_t i = 0; i < sliceCount; ++i)
    {
        const auto &mask = sliceMasks[i];
        const int L = (int)mask.size();
        h_sliceLen[i] = L;

        // positions for this slice
        h_posOffset[i] = h_posIdx_flat.size();
        for (auto p : mask)
            h_posIdx_flat.push_back(p);

        // sublist count in this slice
        const size_t subCount = 1ULL << (2 * L);
        h_sliceSizesOffset[i] = sizesCur;

        // per-slice prefix sums
        h_prefixOffset[i] = prefixCur;
        uint64_t acc = 0;
        for (size_t j = 0; j < subCount; ++j)
        {
            h_prefix_flat.push_back((uint32_t)acc);
            acc += allSlicelistSizes[sizesCur + j];
        }
        prefixCur += subCount;
        sizesCur += subCount;

        // base offset into allSignatures
        h_sliceBaseOffset[i] = sigCur;
        sigCur += (size_t)acc;
    }
    h_posOffset[sliceCount] = h_posIdx_flat.size();
    h_prefixOffset[sliceCount] = prefixCur;

    auto t_load_end = std::chrono::high_resolution_clock::now();

    // ---------------------------
    // 3) LOAD QUERIES (CPU)
    // ---------------------------
    auto t_query_start = std::chrono::high_resolution_clock::now();

    size_t seqLineLength = seqLength + 1; // 20 + '\n'
    std::filesystem::path qp(queryPath);
    size_t fileSize = std::filesystem::file_size(qp);
    if (fileSize % seqLineLength != 0)
    {
        std::fprintf(stderr, "Error: query file not multiple of expected line length (%zu)\n", seqLineLength);
        return 1;
    }
    int queryCount = (int)(fileSize / seqLineLength);

    FILE *qfp = std::fopen(queryPath, "rb");
    if (!qfp)
    {
        std::fprintf(stderr, "Failed to open query file\n");
        return 1;
    }
    std::vector<char> queryDataSet(fileSize);
    if (std::fread(queryDataSet.data(), fileSize, 1, qfp) < 1)
    {
        std::fprintf(stderr, "Failed to read query file\n");
        return 1;
    }
    std::fclose(qfp);

    auto t_query_end = std::chrono::high_resolution_clock::now();
    // ---------------------------
    // 4) GPU ENCODE (replace omp1)
    // ---------------------------
    std::vector<uint64_t> querySignatures(queryCount);
    auto t_enc0 = std::chrono::high_resolution_clock::now();
    gpu_encode_sequences(queryDataSet.data(), queryCount, (int)seqLineLength, (int)seqLength, querySignatures.data());
    auto t_enc1 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "GPU encode done: %d guides in %.3f ms\n",
                 queryCount, std::chrono::duration<double, std::milli>(t_enc1 - t_enc0).count());

    if (queryCount > 0)
    {
        std::printf("First signature (hex): 0x%016llx\n",
                    (unsigned long long)querySignatures[0]);
    }

    // ---- Parity check against a tiny CPU encoder ----
    auto cpu_encode_line = [seqLength](const char *s) -> uint64_t
    {
        auto lut = [](char c) -> uint64_t
        {
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
        };
        uint64_t sig = 0;
        for (size_t j = 0; j < seqLength; ++j) // use seqLength, not 20
            sig |= (lut(s[j]) << (j * 2));
        return sig;
    };

    size_t mismatches = 0;
    for (int i = 0; i < queryCount; ++i)
    {
        const char *line = &queryDataSet[i * (int)seqLineLength];
        if (cpu_encode_line(line) != querySignatures[i])
        {
            ++mismatches;
            if (mismatches < 5)
                std::fprintf(stderr, "Encode mismatch at %d\n", i);
        }
    }
    std::fprintf(stderr, "Encode parity: %zu mismatches out of %d\n", mismatches, queryCount);

    // ---------------------------
    // 5) GPU distance scan
    // ---------------------------
    auto t_scan0 = std::chrono::high_resolution_clock::now();
    std::vector<Hit> hits;
    gpu_distance_scan_flat(
        querySignatures, offtargets, allSignatures, allSlicelistSizes,
        h_sliceLen, h_sliceSizesOffset, h_sliceBaseOffset,
        h_prefix_flat, h_prefixOffset,
        h_posIdx_flat, h_posOffset,
        maxDist, hits);
    auto t_scan1 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "GPU distance hits: %zu (scan %.3f ms)\n",
                 hits.size(), std::chrono::duration<double, std::milli>(t_scan1 - t_scan0).count());

    // ---------------------------
    // 6) CPU reduce + MIT/CFD
    // ---------------------------
    auto t_reduce0 = std::chrono::high_resolution_clock::now();

    // per-query outputs
    std::vector<double> querySignatureMitScores(queryCount, 0.0);
    std::vector<double> querySignatureCfdScores(queryCount, 0.0);
    std::vector<std::atomic<uint64_t>> offtargetsPerQuery(queryCount);
    for (auto &a : offtargetsPerQuery)
        a.store(0);

    // === Bucketize hits by query (O(H), no sort) ===
    // Build per-query counts
    std::vector<size_t> qCount(queryCount, 0);
    for (const auto &h : hits)
        ++qCount[h.q];

    // Prefix offsets
    std::vector<size_t> qOffset(queryCount + 1, 0);
    for (int q = 0; q < queryCount; ++q)
        qOffset[q + 1] = qOffset[q] + qCount[q];

    // Scatter to buckets (stable by arrival)
    std::vector<Hit> hitsByQ(hits.size());
    std::vector<size_t> cursor = qOffset; // working write cursors
    for (const auto &h : hits)
        hitsByQ[cursor[h.q]++] = h;

    // === Reduce per query (same logic as CPU) ===
    const uint64_t numOfftargetToggles =
        (offtargetsCount / ((size_t)sizeof(uint64_t) * (size_t)CHAR_BIT)) + 1ULL;

    // Global bitmap across all queries (for the “Distinct off-targets” stat)
    std::vector<std::atomic<uint64_t>> seenBitmap(numOfftargetToggles);
    for (auto &w : seenBitmap)
        w.store(0, std::memory_order_relaxed);

    uint64_t totalOfftargetsScored = 0;

    for (int q = 0; q < queryCount; ++q)
    {
        const size_t begin = qOffset[q], end = qOffset[q + 1];

        // no hits for this query
        if (begin == end)
        {
            querySignatureMitScores[q] = 10000.0 / (100.0 + 0.0);
            querySignatureCfdScores[q] = 10000.0 / (100.0 + 0.0);
            continue;
        }

        // local dedup bitmap (same shape as CPU)
        std::vector<uint64_t> seenLocal(numOfftargetToggles, 0);
        uint64_t *seenTail = seenLocal.data() + numOfftargetToggles - 1;

        const uint64_t searchSig = querySignatures[q];
        double totMit = 0.0, totCfd = 0.0;
        uint64_t localOfftargetCount = 0;

        // CPU early-exit threshold logic
        const double maximum_sum = (10000.0 - threshold * 100) / threshold;

        for (size_t i = begin; i < end; ++i)
        {
            const auto &h = hitsByQ[i];

            // per-query dedup on id
            uint64_t *word = (seenTail - (h.id / 64));
            const uint64_t bit = 1ULL << (h.id % 64);
            if (*word & bit)
                continue; // already seen this id for this query
            *word |= bit;

            const int dist = popcount64_host(h.mismatches);
            if (dist > maxDist)
                continue;

            localOfftargetCount += h.occ;

            // mark globally seen to compute “Distinct off-targets”
            const size_t gword = h.id / 64;
            const uint64_t gbit = 1ULL << (h.id % 64);
            seenBitmap[gword].fetch_or(gbit, std::memory_order_relaxed);

            // --- MIT (same as CPU) ---
            if (calcMit && dist > 0)
            {
                auto it = precalculatedMITScores.find(h.mismatches);
                if (it != precalculatedMITScores.end())
                    totMit += it->second * (double)h.occ;
            }

            // --- CFD (same as CPU) ---
            if (calcCfd)
            {
                double cfd = 0.0;
                if (dist == 0)
                {
                    cfd = 1.0;
                }
                else
                {
                    cfd = cfdPamPenalties[0b1010]; // PAM NGG (matches CPU)
                    for (size_t pos = 0; pos < 20; ++pos)
                    {
                        size_t mask = pos << 4;
                        uint64_t spos = (searchSig >> (pos * 2)) & 3ULL;
                        spos <<= 2;
                        uint64_t opos = (offtargets[h.id] >> (pos * 2)) & 3ULL;
                        mask |= spos | (opos ^ 3ULL);
                        if ((spos >> 2) != opos)
                            cfd *= cfdPosPenalties[mask];
                    }
                }
                totCfd += cfd * (double)h.occ;
            }

            // --- Early-exit (identical logic to CPU) ---
            if (scoreMethod == "and")
            {
                if (totMit > maximum_sum && totCfd > maximum_sum)
                    break;
            }
            else if (scoreMethod == "or")
            {
                if (totMit > maximum_sum || totCfd > maximum_sum)
                    break;
            }
            else if (scoreMethod == "avg")
            {
                if (((totMit + totCfd) / 2.0) > maximum_sum)
                    break;
            }
            else if (scoreMethod == "mit")
            {
                if (totMit > maximum_sum)
                    break;
            }
            else if (scoreMethod == "cfd")
            {
                if (totCfd > maximum_sum)
                    break;
            }
        }

        offtargetsPerQuery[q].store(localOfftargetCount);
        totalOfftargetsScored += localOfftargetCount;

        querySignatureMitScores[q] = 10000.0 / (100.0 + totMit);
        querySignatureCfdScores[q] = 10000.0 / (100.0 + totCfd);
    }

    auto t_reduce1 = std::chrono::high_resolution_clock::now();

    // Distinct off-targets across all queries
    uint64_t distinctOfftargets = 0;
    for (const auto &w : seenBitmap)
        distinctOfftargets += popcount64_host(w.load(std::memory_order_relaxed));

    // ---------------------------
    // 7) Write results like CPU (but under results\gpu\)
    // ---------------------------
    auto t_out0 = std::chrono::high_resolution_clock::now();

    std::filesystem::create_directories(".\\results\\gpu");
    std::string outName = wantOutputName ? std::string(argv[6]) : "gpu_results.txt";
    std::string outPath = ".\\results\\gpu\\" + outName;

    FILE *out = std::fopen(outPath.c_str(), "w");
    if (!out)
    {
        std::fprintf(stderr, "Error: Could not open output file: %s\n", outPath.c_str());
        return 1;
    }
    std::fprintf(out, "# Query\tMIT_Score\tCFD_Score\tOfftarget_Count\n");
    for (int q = 0; q < queryCount; ++q)
    {
        // For exact parity with CPU, write the sequence (not just index)
        const std::string seq = signatureToSequence(querySignatures[q], (int)seqLength);
        std::fprintf(out, "%s\t", seq.c_str());

        if (calcMit)
            std::fprintf(out, "%f\t", querySignatureMitScores[q]);
        else
            std::fprintf(out, "-1\t");

        if (calcCfd)
            std::fprintf(out, "%f\t", querySignatureCfdScores[q]);
        else
            std::fprintf(out, "-1\t");

        std::fprintf(out, "%llu\n", (unsigned long long)offtargetsPerQuery[q].load());
    }
    std::fclose(out);
    std::fprintf(stderr, "GPU detailed results saved to: %s\n", outPath.c_str());

    auto t_out1 = std::chrono::high_resolution_clock::now();
    auto t_total_end = std::chrono::high_resolution_clock::now();

    // ---------------------------
    // 8) Print CPU-style summary
    // ---------------------------
    const auto ms_load = std::chrono::duration_cast<std::chrono::milliseconds>(t_load_end - t_load_start).count();
    const auto ms_query = std::chrono::duration_cast<std::chrono::milliseconds>(t_query_end - t_query_start).count();
    const auto ms_encode = std::chrono::duration_cast<std::chrono::milliseconds>(t_enc1 - t_enc0).count();
    const auto ms_scan = std::chrono::duration_cast<std::chrono::milliseconds>(t_scan1 - t_scan0).count();
    const auto ms_reduce = std::chrono::duration_cast<std::chrono::milliseconds>(t_reduce1 - t_reduce0).count();
    const auto ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(t_out1 - t_out0).count();
    const auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(t_total_end - t_total_start).count();

    std::fprintf(stderr, "\n=== GPU EXECUTION SUMMARY ===\n");
    std::fprintf(stderr, "Total queries processed: %d\n", queryCount);
    std::fprintf(stderr, "Total off-targets scored: %llu\n", (unsigned long long)totalOfftargetsScored);
    std::fprintf(stderr, "Average off-targets per query: %.2f\n",
                 queryCount > 0 ? (double)totalOfftargetsScored / (double)queryCount : 0.0);
    std::fprintf(stderr, "Loading time: %lld ms (index)\n", (long long)ms_load);
    std::fprintf(stderr, "Query load time: %lld ms\n", (long long)ms_query);
    std::fprintf(stderr, "Encode (GPU) time: %lld ms\n", (long long)ms_encode);
    std::fprintf(stderr, "Distance scan (GPU) time: %lld ms\n", (long long)ms_scan);
    std::fprintf(stderr, "Reduce+Score (CPU) time: %lld ms\n", (long long)ms_reduce);
    std::fprintf(stderr, "Output time: %lld ms\n", (long long)ms_out);
    std::fprintf(stderr, "Total execution time: %lld ms\n", (long long)ms_total);

    // Two helpful rates:
    std::fprintf(stderr, "Processing rate (Reduce only): %.2f queries/sec\n",
                 ms_reduce > 0 ? (queryCount * 1000.0) / (double)ms_reduce : 0.0);
    std::fprintf(stderr, "Processing rate (Scan+Reduce): %.2f queries/sec\n",
                 (ms_scan + ms_reduce) > 0 ? (queryCount * 1000.0) / (double)(ms_scan + ms_reduce) : 0.0);

    std::fprintf(stderr, "Off-target scoring rate: %.2f off-targets/sec\n",
                 ms_reduce > 0 ? (totalOfftargetsScored * 1000.0) / (double)ms_reduce : 0.0);
    std::fprintf(stderr, "Distinct off-targets: %llu\n", (unsigned long long)distinctOfftargets);
    std::fprintf(stderr, "=============================\n\n");

    return 0;
}