#pragma once
#include <cstdint>
#include <unordered_map>
#include <array>

// MIT: mismatch-mask -> contribution
extern std::unordered_map<uint64_t, double> precalculatedMITScores;

// CFD: simple arrays
extern std::array<double, 16> cfdPamPenalties;    // index like 0b1010 for NGG
extern std::array<double,  (20u*16u)> cfdPosPenalties; // indexed by: (pos<<4)|(sg<<2)|(rc(off))
