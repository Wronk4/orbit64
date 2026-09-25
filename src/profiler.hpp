#pragma once
// Minimal sampling profiler for headless runs (`--profile <file>`): a
// background thread periodically suspends the thread that started it and
// records its instruction pointer plus the call chain above it (unwound
// with the executable's unwind tables).
// Only raw addresses are written; tools/prof_report.py turns them into
// flat/inclusive per-function tables with nm. Windows x64 only; elsewhere
// use perf / Instruments instead.

#include <string>

namespace profiler {

// Starts sampling the calling thread. Returns false if unsupported.
bool start(const std::string& out_path, unsigned interval_us = 1000);
// Stops sampling and writes the samples file.
void stop();

} // namespace profiler
