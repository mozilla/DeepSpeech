#include "util/parallel_read.hh"

#include "util/file.hh"

#ifdef WITH_THREADS
#include "util/thread_pool.hh"

namespace util {
namespace {

class Reader {
  public:
    explicit Reader(int fd) : fd_(fd) {}

    struct Request {
      void *to;
      std::size_t size;
      uint64_t offset;

      bool operator==(const Request &other) const {
        return (to == other.to) && (size == other.size) && (offset == other.offset);
      }
    };

    void operator()(const Request &request) {
      util::ErsatzPRead(fd_, request.to, request.size, request.offset);
    }

  private:
    int fd_;
};

} // namespace

void ParallelRead(int fd, void *to, std::size_t amount, uint64_t offset) {
  Reader::Request poison;
  poison.to = NULL;
  poison.size = 0;
  poison.offset = 0;
  unsigned threads = boost::thread::hardware_concurrency();
  if (!threads) threads = 2;
  ThreadPool<Reader> pool(2 /* don't need much of a queue */, threads, fd, poison);
  const std::size_t kBatch = 1ULL << 25; // 32 MB
  Reader::Request request;
  request.to = to;
  request.size = kBatch;
  request.offset = offset;
  for (; amount > kBatch; amount -= kBatch) {
    pool.Produce(request);
    request.to = reinterpret_cast<uint8_t*>(request.to) + kBatch;
    request.offset += kBatch;
  }
  request.size = amount;
  if (request.size) {
    pool.Produce(request);
  }
}

} // namespace util

#else // WITH_THREADS

namespace util {
void ParallelRead(int fd, void *to, std::size_t amount, uint64_t offset) {
 util::ErsatzPRead(fd, to, amount, offset);
}
} // namespace util

#endif
