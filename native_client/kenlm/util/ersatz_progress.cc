#include "util/ersatz_progress.hh"

#include <algorithm>
#include <ostream>
#include <limits>
#include <string>

namespace util {

namespace { const unsigned char kWidth = 100; }

const char kProgressBanner[] = "----5---10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---95--100\n";

ErsatzProgress::ErsatzProgress() : current_(0), next_(std::numeric_limits<uint64_t>::max()), complete_(next_), out_(NULL) {}

ErsatzProgress::~ErsatzProgress() {
  if (out_) Finished();
}

ErsatzProgress::ErsatzProgress(uint64_t complete, std::ostream *to, const std::string &message)
  : current_(0), next_(complete / kWidth), complete_(complete), stones_written_(0), out_(to) {
  if (!out_) {
    next_ = std::numeric_limits<uint64_t>::max();
    return;
  }
  if (!message.empty()) *out_ << message << '\n';
  *out_ << kProgressBanner;
}

void ErsatzProgress::Milestone() {
  if (!out_) { current_ = 0; return; }
  if (!complete_) return;
  unsigned char stone = std::min(static_cast<uint64_t>(kWidth), (current_ * kWidth) / complete_);

  for (; stones_written_ < stone; ++stones_written_) {
    (*out_) << '*';
  }
  if (stone == kWidth) {
    (*out_) << std::endl;
    next_ = std::numeric_limits<uint64_t>::max();
    out_ = NULL;
  } else {
    next_ = std::max(next_, ((stone + 1) * complete_ + kWidth - 1) / kWidth);
  }
}

} // namespace util
