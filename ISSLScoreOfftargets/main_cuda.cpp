#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <filesystem>
#include <chrono>
#include <string>
#include "issl_cuda.cuh"
// DO NOT include ISSLScoreOfftargets.hpp yet (it pulls Boost and CPU-only deps)

int main(int argc, char** argv){
    if (argc < 6){
        std::fprintf(stderr,
          "Usage: %s [issltable] [query file] [max distance] [score-threshold] [score-method] [output-file-optional]\n",
          argv[0]);
        return 1;
    }

    const char* isslPath   = argv[1];
    const char* queryPath  = argv[2];

    // 1) Read ISSL header to get seqLength (no Boost needed)
    FILE* isslFp = std::fopen(isslPath, "rb");
    if (!isslFp){ std::fprintf(stderr, "Error: could not open ISSL file\n"); return 1; }
    std::vector<size_t> hdr(3);
    if (std::fread(hdr.data(), sizeof(size_t), hdr.size(), isslFp) == 0){
        std::fprintf(stderr, "Error: invalid ISSL header\n"); return 1;
    }
    size_t /*offtargetsCount=*/ hdr0 = hdr[0];
    size_t seqLength = hdr[1];   // typically 20
    size_t /*sliceCount=*/ hdr2 = hdr[2];
    std::fclose(isslFp);

    // 2) Load queries into flat char buffer
    size_t seqLineLength = seqLength + 1; // data lines are 20 chars + '\n'
    std::filesystem::path qp(queryPath);
    size_t fileSize = std::filesystem::file_size(qp);
    if (fileSize % seqLineLength != 0){
        std::fprintf(stderr, "Error: query file not multiple of expected line length (%zu)\n", seqLineLength);
        return 1;
    }
    int queryCount = (int)(fileSize / seqLineLength);

    FILE* qfp = std::fopen(queryPath, "rb");
    if (!qfp){ std::fprintf(stderr, "Failed to open query file\n"); return 1; }
    std::vector<char> queryDataSet(fileSize);
    if (std::fread(queryDataSet.data(), fileSize, 1, qfp) < 1){
        std::fprintf(stderr, "Failed to read query file\n"); return 1;
    }
    std::fclose(qfp);

    // 3) GPU encode (replaces your OpenMP encode block)
    std::vector<uint64_t> querySignatures(queryCount);
    auto t0 = std::chrono::high_resolution_clock::now();
    gpu_encode_sequences(queryDataSet.data(), queryCount, (int)seqLineLength, querySignatures.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "GPU encode done: %d guides in %.3f ms\n", queryCount, enc_ms);

    // Sanity: print first signature
    if (queryCount > 0){
        std::printf("First signature (hex): 0x%016llx\n",
            (unsigned long long)querySignatures[0]);
    }

    // NEXT STEP (not compiled yet): call a GPU distance kernel, then reuse your CPU MIT/CFD.
    return 0;
}
