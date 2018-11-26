#ifndef UTIL_PARALLEL_READ__
#define UTIL_PARALLEL_READ__

/* Read pieces of a file in parallel.  This has a very specific use case:
 * reading files from Lustre is CPU bound so multiple threads actually
 * increases throughput.  Speed matters when an LM takes a terabyte.
 */

#include <cstddef>
#include <stdint.h>

namespace util {
void ParallelRead(int fd, void *to, std::size_t amount, uint64_t offset);
} // namespace util

#endif // UTIL_PARALLEL_READ__
