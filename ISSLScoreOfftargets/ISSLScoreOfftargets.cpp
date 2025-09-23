#include "ISSLScoreOfftargets.hpp"
#include <atomic>

using std::cout;
using std::endl;
using std::pair;
using std::string;
using std::unordered_map;
using std::vector;

// Char to binary encoding
const vector<uint8_t> nucleotideIndex{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3};
// Binary to char encoding
const vector<char> signatureIndex{'A', 'C', 'G', 'T'};

uint64_t sequenceToSignature(const std::string &seq, uint64_t seqLen) {
  uint64_t signature = 0;
  for (uint64_t j = 0; j < seqLen; j++) {
    signature |= static_cast<uint64_t>(nucleotideIndex[seq[j]]) << (j * 2);
  }
  return signature;
}

string signatureToSequence(uint64_t sig, uint64_t seqLen) {
  string sequence = string(seqLen, ' ');
  for (uint64_t j = 0; j < seqLen; j++) {
    sequence[j] = signatureIndex[(sig >> (j * 2)) & 0x3];
  }
  return sequence;
}

int main(int argc, char **argv) {

  // Start the total timer
  auto t_total_start = std::chrono::high_resolution_clock::now();

  // Start the timer for loading the index
  auto startLoading = std::chrono::high_resolution_clock::now();

  if (argc < 4) {
    fprintf(stderr,
            "Usage: %s [issltable] [query file] [max distance] "
            "[score-threshold] [score-method] [output-file-optional]\n",
            argv[0]);
    exit(1);
  }

  /** The maximum number of mismatches */
  int maxDist = atoi(argv[3]);

  /** The threshold used to exit scoring early */
  double threshold = atof(argv[4]);

  /** Scoring methods. To exit early:
   *      - only CFD must drop below `threshold`
   *      - only MIT must drop below `threshold`
   *      - both CFD and MIT must drop below `threshold`
   *      - CFD or MIT must drop below `threshold`
   *      - the average of CFD and MIT must below `threshold`
   */
  string argScoreMethod = argv[5];
  otScoreMethod scoreMethod;
  bool calcCfd = false;
  bool calcMit = false;
  if (!argScoreMethod.compare("and")) {
    scoreMethod = otScoreMethod::mitAndCfd;
    calcCfd = true;
    calcMit = true;
  } else if (!argScoreMethod.compare("or")) {
    scoreMethod = otScoreMethod::mitOrCfd;
    calcCfd = true;
    calcMit = true;
  } else if (!argScoreMethod.compare("avg")) {
    scoreMethod = otScoreMethod::avgMitCfd;
    calcCfd = true;
    calcMit = true;
  } else if (!argScoreMethod.compare("mit")) {
    scoreMethod = otScoreMethod::mit;
    calcMit = true;
  } else if (!argScoreMethod.compare("cfd")) {
    scoreMethod = otScoreMethod::cfd;
    calcCfd = true;
  } else {
    fprintf(stderr, "Invalid scoring method. Acceptable options are: 'and', "
                    "'or', 'avg', 'mit', 'cfd'");
    exit(1);
  }

  /** Begin reading the binary encoded ISSL, structured as:
   *  - The header (3 items)
   *  - All binary-encoded off-target sites
   *  - Slice masks
   *  - Size of slice 1 lists
   *  - Contents of slice 1 lists
   *  ...
   *  - Size of slice N lists (N being the number of slices)
   *  - Contents of slice N lists
   */
  FILE *isslFp = fopen(argv[1], "rb");

  if (isslFp == NULL) {
    throw std::runtime_error("Error reading index: could not open file\n");
  }

  /** The index contains a fixed-sized header
   *      - the number of unique off-targets in the index
   *      - the length of an off-target
   *      - the number of slices
   */
  vector<size_t> slicelistHeader(3);

  if (fread(slicelistHeader.data(), sizeof(size_t), slicelistHeader.size(),
            isslFp) == 0) {
    throw std::runtime_error("Error reading index: header invalid\n");
  }

  size_t offtargetsCount = slicelistHeader[0];
  size_t seqLength = slicelistHeader[1];
  size_t sliceCount = slicelistHeader[2];

  /** Load in all of the off-target sites */
  vector<uint64_t> offtargets(offtargetsCount);
  if (fread(offtargets.data(), sizeof(uint64_t), offtargetsCount, isslFp) ==
      0) {
    throw std::runtime_error(
        "Error reading index: loading off-target sequences failed\n");
  }

  /** Read the slice masks and generate 2 bit masks */
  vector<vector<uint64_t>> sliceMasks;
  for (size_t i = 0; i < sliceCount; i++) {
    uint64_t maskBinary;
    fread(&maskBinary, sizeof(uint64_t), 1, isslFp);

    vector<uint64_t> mask;
    for (uint64_t j = 0; j < seqLength; j++) {
      if (maskBinary & (1ULL << j)) {
        mask.push_back(j);
      }
    }
    sliceMasks.push_back(mask);
  }

  /** Calculate the total number of lists based on slice count and width */
  size_t sliceListCount = 0;
  for (size_t i = 0; i < sliceCount; i++) {
    sliceListCount += 1ULL << (sliceMasks[i].size() * 2);
  }

  /** The number of signatures embedded per slice. Store continguously */
  vector<size_t> allSlicelistSizes(sliceListCount);

  /** The contents of the slices. Stored contiguously
   *  Each signature (64-bit) is structured as:
   *      <occurrences 32-bit><off-target-id 32-bit>
   */
  vector<uint64_t> allSignatures(offtargetsCount * sliceCount);

  /** The number of signatures embedded per slice. Store continguously */
  sliceListCount = 0;
  for (size_t i = 0; i < sliceCount; i++) {
    size_t sliceListSize = 1ULL << (sliceMasks[i].size() * 2);
    if (fread(allSlicelistSizes.data() + sliceListCount, sizeof(size_t),
              sliceListSize, isslFp) == 0) {
      throw std::runtime_error(
          "Error reading index: reading slice list sizes failed\n");
    }

    if (fread(allSignatures.data() + (offtargetsCount * i), sizeof(uint64_t),
              offtargetsCount, isslFp) == 0) {
      throw std::runtime_error(
          "Error reading index: reading slice contents failed\n");
    }

    sliceListCount += 1ULL << (sliceMasks[i].size() * 2);
  }

  /** End reading the index */
  fclose(isslFp);

  auto t_index_end = std::chrono::high_resolution_clock::now();
  std::cout << "Index loading time: "
            << std::chrono::duration<double>(t_index_end - startLoading).count()
            << " seconds" << std::endl;

  // Start query loading timer
  auto t_query_start = std::chrono::high_resolution_clock::now();

  /** Prevent assessing an off-target site for multiple slices
   *
   *      Create enough 1-bit "seen" flags for the off-targets
   *      We only want to score a candidate guide against an off-target once.
   *      The least-significant bit represents the first off-target
   *      0 0 0 1   0 1 0 0   would indicate that the 3rd and 5th off-target
   * have been seen. The CHAR_BIT macro tells us how many bits are in a byte
   * (C++ >= 8 bits per byte)
   */
  uint64_t numOfftargetToggles =
      (offtargetsCount / ((size_t)sizeof(uint64_t) * (size_t)CHAR_BIT)) + 1;
  // Global bitmap (thread-safe) for distinct off-target IDs seen across ALL
  // queries
  std::vector<std::atomic<uint64_t>> seenBitmap(numOfftargetToggles);
  for (auto &w : seenBitmap)
    w.store(0, std::memory_order_relaxed);

  /** Start constructing index in memory
   *
   *      To begin, reverse the contiguous storage of the slices,
   *         into the following:
   *
   *         + Slice 0 :
   *         |---- AAAA : <slice contents>
   *         |---- AAAC : <slice contents>
   *         |----  ...
   *         |
   *         + Slice 1 :
   *         |---- AAAA : <slice contents>
   *         |---- AAAC : <slice contents>
   *         |---- ...
   *         | ...
   */

  vector<vector<uint64_t *>> sliceLists(sliceCount);
  // Assign sliceLists size based on each slice length
  for (size_t i = 0; i < sliceCount; i++) {
    sliceLists[i] = vector<uint64_t *>(1ULL << (sliceMasks[i].size() * 2));
  }

  uint64_t *offset = allSignatures.data();
  size_t sliceLimitOffset = 0;
  for (size_t i = 0; i < sliceCount; i++) {
    size_t sliceLimit = 1ULL << (sliceMasks[i].size() * 2);
    for (size_t j = 0; j < sliceLimit; j++) {
      size_t idx = sliceLimitOffset + j;
      sliceLists[i][j] = offset;
      offset += allSlicelistSizes[idx];
    }
    sliceLimitOffset += sliceLimit;
  }

  auto endLoading = std::chrono::high_resolution_clock::now();
  auto startProcessing = std::chrono::high_resolution_clock::now();

  // TODO: rewrite
  /** Load query file (candidate guides)
   *      and prepare memory for calculated global scores
   */
  size_t seqLineLength = seqLength + 1;
  std::filesystem::path queryFile(argv[2]);
  size_t fileSize = std::filesystem::file_size(queryFile);
  if (fileSize % seqLineLength != 0) {
    fprintf(stderr,
            "Error: query file is not a multiple of the expected line length "
            "(%zu)\n",
            seqLineLength);
    fprintf(stderr, "The sequence length may be incorrect; alternatively, the "
                    "line endings\n");
    fprintf(stderr, "may be something other than LF, or there may be junk at "
                    "the end of the file.\n");
    exit(1);
  }
  size_t queryCount = fileSize / seqLineLength;
  FILE *fp = fopen(argv[2], "rb");
  vector<char> queryDataSet(fileSize);
  vector<uint64_t> querySignatures(queryCount);
  vector<double> querySignatureMitScores(queryCount);
  vector<double> querySignatureCfdScores(queryCount);

  if (fread(queryDataSet.data(), fileSize, 1, fp) < 1) {
    fprintf(stderr, "Failed to read in query file.\n");
    exit(1);
  }
  fclose(fp);

/** Binary encode query sequences */
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < queryCount; i++) {
      char *ptr = &queryDataSet[i * seqLineLength];
      uint64_t signature = sequenceToSignature(ptr, 20);
      querySignatures[i] = signature;
    }
  }

  auto t_query_end = std::chrono::high_resolution_clock::now();
  std::cout
      << "Query loading time: "
      << std::chrono::duration<double>(t_query_end - t_query_start).count()
      << " seconds" << std::endl;
  // Start ISSL scoring timer
  auto t_issl_start = std::chrono::high_resolution_clock::now();

  /** Begin scoring */
  // Add counters for statistics
  std::atomic<uint64_t> totalOfftargetsScored{0};
  std::atomic<uint64_t> totalQueriesProcessed{0};
  std::vector<std::atomic<uint64_t>> offtargetsPerQuery(queryCount);
  for (auto &counter : offtargetsPerQuery) {
    counter.store(0);
  }

// Main omp 2 (Scoring loop)
#pragma omp parallel
  {
    vector<uint64_t> offtargetToggles(numOfftargetToggles);
    uint64_t *offtargetTogglesTail =
        offtargetToggles.data() + numOfftargetToggles - 1;
/** For each candidate guide */
// TODO: update to openMP > v2 (Use clang compiler)
#pragma omp for
    for (int searchIdx = 0; searchIdx < querySignatures.size(); searchIdx++) {

      auto searchSignature = querySignatures[searchIdx];
      uint64_t localOfftargetCount = 0; // Local counter for this query

      /** Global scores */
      double totScoreMit = 0.0;
      double totScoreCfd = 0.0;

      double maximum_sum = (10000.0 - threshold * 100) / threshold;
      bool checkNextSlice = true;

      size_t sliceLimitOffset = 0;
      /** For each ISSL slice */
      for (size_t i = 0; i < sliceCount; i++) {
        vector<uint64_t> &sliceMask = sliceMasks[i];
        auto &sliceList = sliceLists[i];

        uint64_t searchSlice = 0ULL;
        for (int j = 0; j < sliceMask.size(); j++) {
          searchSlice |= ((searchSignature >> (sliceMask[j] * 2)) & 3ULL)
                         << (j * 2);
        }

        size_t idx = sliceLimitOffset + searchSlice;

        size_t signaturesInSlice = allSlicelistSizes[idx];
        uint64_t *sliceOffset = sliceList[searchSlice];

        /** For each off-target signature in slice */
        for (size_t j = 0; j < signaturesInSlice; j++) {
          auto signatureWithOccurrencesAndId = sliceOffset[j];
          auto signatureId = signatureWithOccurrencesAndId & 0xFFFFFFFFULL;
          uint32_t occurrences = (signatureWithOccurrencesAndId >> (32));

          /** Prevent assessing the same off-target for multiple slices */
          uint64_t seenOfftargetAlready = 0;
          uint64_t *ptrOfftargetFlag =
              (offtargetTogglesTail - (signatureId / 64));
          seenOfftargetAlready =
              (*ptrOfftargetFlag >> (signatureId % 64)) & 1ULL;

          if (!seenOfftargetAlready) {
            *ptrOfftargetFlag |= (1ULL << (signatureId % 64));

            /** Find the positions of mismatches */
            uint64_t xoredSignatures =
                searchSignature ^ offtargets[signatureId];
            uint64_t evenBits = xoredSignatures & 0xAAAAAAAAAAAAAAAAULL;
            uint64_t oddBits = xoredSignatures & 0x5555555555555555ULL;
            uint64_t mismatches = (evenBits >> 1) | oddBits;
            uint64_t dist = popcount64(mismatches);

            if (dist >= 0 && dist <= 4) {
              // Count this off-target
              localOfftargetCount += occurrences;

              // --- NEW: mark this signatureId as seen globally (distinct
              // sites)
              size_t word = signatureId / 64;
              uint64_t bit = 1ULL << (signatureId % 64);
              seenBitmap[word].fetch_or(bit, std::memory_order_relaxed);
              // --- END NEW

              // Begin calculating MIT score
              if (calcMit) {
                if (dist > 0) {
                  totScoreMit += precalculatedMITScores.at(mismatches) *
                                 (double)occurrences;
                }
              }

              // Begin calculating CFD score
              if (calcCfd) {
                /** "In other words, for the CFD score, a value of 0
                 *      indicates no predicted off-target activity whereas
                 *      a value of 1 indicates a perfect match"
                 *      John Doench, 2016.
                 *      https://www.nature.com/articles/nbt.3437
                 */
                double cfdScore = 0;
                if (dist == 0) {
                  cfdScore = 1;
                } else {
                  cfdScore = cfdPamPenalties[0b1010]; // PAM: NGG, TODO: do not
                                                      // hard-code the PAM

                  for (size_t pos = 0; pos < 20; pos++) {
                    size_t mask = pos << 4;

                    /** Create the mask to look up the position-identity score
                     *      In Python... c2b is char to bit
                     *       mask = pos << 4
                     *       mask |= c2b[sgRNA[pos]] << 2
                     *       mask |= c2b[revcom(offTaret[pos])]
                     *
                     *      Find identity at `pos` for search signature
                     *      example: find identity in pos=2
                     *       Recall ISSL is inverted, hence:
                     *                   3'-  T  G  C  C  G  A -5'
                     *       start           11 10 01 01 10 00
                     *       3UL << pos*2    00 00 00 11 00 00
                     *       and             00 00 00 01 00 00
                     *       shift           00 00 00 00 01 00
                     */
                    uint64_t searchSigIdentityPos = searchSignature;
                    searchSigIdentityPos &= (3ULL << (pos * 2));
                    searchSigIdentityPos = searchSigIdentityPos >> (pos * 2);
                    searchSigIdentityPos = searchSigIdentityPos << 2;

                    /** Find identity at `pos` for offtarget
                     *      Example: find identity in pos=2
                     *      Recall ISSL is inverted, hence:
                     *                  3'-  T  G  C  C  G  A -5'
                     *      start           11 10 01 01 10 00
                     *      3UL<<pos*2      00 00 00 11 00 00
                     *      and             00 00 00 01 00 00
                     *      shift           00 00 00 00 00 01
                     *      rev comp 3UL    00 00 00 00 00 10 (done below)
                     */
                    uint64_t offtargetIdentityPos = offtargets[signatureId];
                    offtargetIdentityPos &= (3ULL << (pos * 2));
                    offtargetIdentityPos = offtargetIdentityPos >> (pos * 2);

                    /** Complete the mask
                     *      reverse complement (^3UL) `offtargetIdentityPos`
                     * here
                     */
                    mask = (mask | searchSigIdentityPos |
                            (offtargetIdentityPos ^ 3UL));

                    if (searchSigIdentityPos >> 2 != offtargetIdentityPos) {
                      cfdScore *= cfdPosPenalties[mask];
                    }
                  }
                }
                totScoreCfd += cfdScore * (double)occurrences;
              }

              /** Stop calculating global score early if possible */
              if (scoreMethod == otScoreMethod::mitAndCfd) {
                if (totScoreMit > maximum_sum && totScoreCfd > maximum_sum) {
                  checkNextSlice = false;
                  break;
                }
              }
              if (scoreMethod == otScoreMethod::mitOrCfd) {
                if (totScoreMit > maximum_sum || totScoreCfd > maximum_sum) {
                  checkNextSlice = false;
                  break;
                }
              }
              if (scoreMethod == otScoreMethod::avgMitCfd) {
                if (((totScoreMit + totScoreCfd) / 2.0) > maximum_sum) {
                  checkNextSlice = false;
                  break;
                }
              }
              if (scoreMethod == otScoreMethod::mit) {
                if (totScoreMit > maximum_sum) {
                  checkNextSlice = false;
                  break;
                }
              }
              if (scoreMethod == otScoreMethod::cfd) {
                if (totScoreCfd > maximum_sum) {
                  checkNextSlice = false;
                  break;
                }
              }
            }
          }
        }
        if (!checkNextSlice)
          break;
        sliceLimitOffset += 1ULL << (sliceMasks[i].size() * 2);
      }

      // Update counters
      totalOfftargetsScored.fetch_add(localOfftargetCount);
      offtargetsPerQuery[searchIdx].store(localOfftargetCount);
      totalQueriesProcessed.fetch_add(1);

      querySignatureMitScores[searchIdx] = 10000.0 / (100.0 + totScoreMit);
      querySignatureCfdScores[searchIdx] = 10000.0 / (100.0 + totScoreCfd);

      memset(offtargetToggles.data(), 0,
             sizeof(uint64_t) * offtargetToggles.size());
    }
  }

  auto t_issl_end = std::chrono::high_resolution_clock::now();
  std::cout << "ISSL scoring time: "
            << std::chrono::duration<double>(t_issl_end - t_issl_start).count()
            << " seconds" << std::endl;

  // Start output timer
  auto t_output_start = std::chrono::high_resolution_clock::now();

  // Calculate timing
  auto endProcessing = std::chrono::high_resolution_clock::now();
  auto loadingTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                         endLoading - startLoading)
                         .count();
  auto processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                            endProcessing - startProcessing)
                            .count();
  auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                       endProcessing - startLoading)
                       .count();

  // Print summary statistics
  fprintf(stderr, "\n=== EXECUTION SUMMARY ===\n");
  fprintf(stderr, "Total queries processed: %lu\n",
          totalQueriesProcessed.load());
  fprintf(stderr, "Total off-targets scored: %lu\n",
          totalOfftargetsScored.load());
  fprintf(stderr, "Average off-targets per query: %.2f\n",
          totalQueriesProcessed.load() > 0
              ? (double)totalOfftargetsScored.load() /
                    totalQueriesProcessed.load()
              : 0.0);
  fprintf(stderr, "Loading time: %ld ms\n", loadingTime);
  fprintf(stderr, "Processing time: %ld ms\n", processingTime);
  fprintf(stderr, "Total execution time: %ld ms\n", totalTime);
  fprintf(stderr, "Processing rate: %.2f queries/sec\n",
          processingTime > 0
              ? (totalQueriesProcessed.load() * 1000.0) / processingTime
              : 0.0);
  fprintf(stderr, "Off-target scoring rate: %.2f off-targets/sec\n",
          processingTime > 0
              ? (totalOfftargetsScored.load() * 1000.0) / processingTime
              : 0.0);
  // Compute distinct off-target count by popcount over the global bitmap
  uint64_t distinctOfftargets = 0;
  for (const auto &w : seenBitmap) {
    distinctOfftargets += popcount64(w.load(std::memory_order_relaxed));
  }
  fprintf(stderr, "Distinct off-targets: %llu\n",
          (unsigned long long)distinctOfftargets);
  fprintf(stderr, "========================\n\n");

  // Always save detailed results to file
  FILE *outputFile = nullptr;

  // Create the results directory if it doesn't exist
  std::string resultsDir =
      "C:\\Users\\ronil\\Documents\\UNI\\Year 4 Sem "
      "1\\EGH400-1\\Code\\CracklingPlusPlus\\build\\results";

  // Create directory if it doesn't exist
  if (!std::filesystem::exists(resultsDir)) {
    if (!std::filesystem::create_directories(resultsDir)) {
      fprintf(stderr, "Warning: Could not create results directory: %s\n",
              resultsDir.c_str());
    }
  }

  // Generate filename based on input file and parameters
  std::string defaultFilename;
  if (argc > 6) {
    defaultFilename = std::string(argv[6]);
  } else {
    // Extract base name from input file
    std::filesystem::path inputPath(argv[2]);
    std::string baseName = inputPath.stem().string();
    defaultFilename = baseName + "_results.txt";
  }

  // Construct full path for output file
  std::string outputPath = resultsDir + "\\" + defaultFilename;

  outputFile = fopen(outputPath.c_str(), "w");
  if (outputFile) {
    fprintf(outputFile, "# Query\tMIT_Score\tCFD_Score\tOfftarget_Count\n");
  } else {
    fprintf(stderr, "Error: Could not open output file: %s\n",
            outputPath.c_str());
  }

  /** Print global scores to stdout */
  for (size_t searchIdx = 0; searchIdx < querySignatures.size(); searchIdx++) {
    auto querySequence = signatureToSequence(querySignatures[searchIdx], 20);
    printf("%s\t", querySequence.c_str());
    if (calcMit)
      printf("%f\t", querySignatureMitScores[searchIdx]);
    else
      printf("-1\t");

    if (calcCfd)
      printf("%f\n", querySignatureCfdScores[searchIdx]);
    else
      printf("-1\n");

    // Write to output file if specified
    if (outputFile) {
      fprintf(outputFile, "%s\t", querySequence.c_str());
      if (calcMit)
        fprintf(outputFile, "%f\t", querySignatureMitScores[searchIdx]);
      else
        fprintf(outputFile, "-1\t");

      if (calcCfd)
        fprintf(outputFile, "%f\t", querySignatureCfdScores[searchIdx]);
      else
        fprintf(outputFile, "-1\t");

      fprintf(outputFile, "%lu\n", offtargetsPerQuery[searchIdx].load());
    }
  }

  auto t_output_end = std::chrono::high_resolution_clock::now();
  std::cout
      << "Output time: "
      << std::chrono::duration<double>(t_output_end - t_output_start).count()
      << " seconds" << std::endl;

  auto t_total_end = std::chrono::high_resolution_clock::now();
  std::cout
      << "Total execution time: "
      << std::chrono::duration<double>(t_total_end - t_total_start).count()
      << " seconds" << std::endl;

  if (outputFile) {
    fclose(outputFile);
    std::string resultsDir =
        "C:\\Users\\ronil\\Documents\\UNI\\Year 4 Sem "
        "1\\EGH400-1\\Code\\CracklingPlusPlus\\build\\results";
    std::string outputPath = resultsDir + "\\" + std::string(argv[6]);
    fprintf(stderr, "Detailed results saved to: %s\n", outputPath.c_str());
  }
}