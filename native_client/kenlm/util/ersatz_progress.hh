#ifndef UTIL_ERSATZ_PROGRESS_H
#define UTIL_ERSATZ_PROGRESS_H

#include <iostream>
#include <string>
#include <stdint.h>

// Ersatz version of boost::progress so core language model doesn't depend on
// boost.  Also adds option to print nothing.

namespace util {

extern const char kProgressBanner[];

class ErsatzProgress {
  public:
    // No output.
    ErsatzProgress();

    // Null means no output.  The null value is useful for passing along the ostream pointer from another caller.
    explicit ErsatzProgress(uint64_t complete, std::ostream *to = &std::cerr, const std::string &message = "");

#if __cplusplus >= 201103L
    ErsatzProgress(ErsatzProgress &&from) noexcept : current_(from.current_), next_(from.next_), complete_(from.complete_), stones_written_(from.stones_written_), out_(from.out_) {
      from.out_ = nullptr;
      from.next_ = (uint64_t)-1;
    }
#endif

    ~ErsatzProgress();

    ErsatzProgress &operator++() {
      if (++current_ >= next_) Milestone();
      return *this;
    }

    ErsatzProgress &operator+=(uint64_t amount) {
      if ((current_ += amount) >= next_) Milestone();
      return *this;
    }

    void Set(uint64_t to) {
      if ((current_ = to) >= next_) Milestone();
    }

    void Finished() {
      Set(complete_);
    }

  private:
    void Milestone();

    uint64_t current_, next_, complete_;
    unsigned char stones_written_;
    std::ostream *out_;

    // noncopyable
    ErsatzProgress(const ErsatzProgress &other);
    ErsatzProgress &operator=(const ErsatzProgress &other);
};

} // namespace util

#endif // UTIL_ERSATZ_PROGRESS_H
