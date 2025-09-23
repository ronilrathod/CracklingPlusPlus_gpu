#include "ot_penalties.hpp"

// Neutral/no-op scoring so the program runs end-to-end.
// Replace with your real values for accuracy parity.

std::unordered_map<uint64_t, double> precalculatedMITScores; // empty => contributes 0

std::array<double, 16> cfdPamPenalties = []{
    std::array<double,16> a{};
    for (auto& x : a) x = 1.0;        // neutral PAM
    return a;
}();

std::array<double, (20u*16u)> cfdPosPenalties = []{
    std::array<double, (20u*16u)> a{};
    for (auto& x : a) x = 1.0;        // neutral per-position penalty
    return a;
}();
