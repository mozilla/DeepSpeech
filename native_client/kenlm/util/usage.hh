#ifndef UTIL_USAGE_H
#define UTIL_USAGE_H
#include <cstddef>
#include <iosfwd>
#include <string>
#include <stdint.h>

namespace util {
// Time in seconds since process started.  Zero on unsupported platforms.
double WallTime();

// User + system time, process-wide.
double CPUTime();

// User + system time, thread-specific.
double ThreadTime();

// Resident usage in bytes.
uint64_t RSSMax();

void PrintUsage(std::ostream &to);

// Determine how much physical memory there is.  Return 0 on failure.
uint64_t GuessPhysicalMemory();

// Parse a size like unix sort.  Sadly, this means the default multiplier is K.
uint64_t ParseSize(const std::string &arg);

} // namespace util
#endif // UTIL_USAGE_H
