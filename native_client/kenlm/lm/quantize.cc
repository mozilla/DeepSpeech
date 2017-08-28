/* Quantize into bins of equal size as described in
 * M. Federico and N. Bertoldi. 2006. How many bits are needed
 * to store probabilities for phrase-based translation? In Proc.
 * of the Workshop on Statistical Machine Translation, pages
 * 94–101, New York City, June. Association for Computa-
 * tional Linguistics.
 */

#include "lm/quantize.hh"

#include "lm/binary_format.hh"
#include "lm/lm_exception.hh"
#include "util/file.hh"

#include <algorithm>
#include <numeric>

namespace lm {
namespace ngram {

namespace {

void MakeBins(std::vector<float> &values, float *centers, uint32_t bins) {
  std::sort(values.begin(), values.end());
  std::vector<float>::const_iterator start = values.begin(), finish;
  for (uint32_t i = 0; i < bins; ++i, ++centers, start = finish) {
    finish = values.begin() + ((values.size() * static_cast<uint64_t>(i + 1)) / bins);
    if (finish == start) {
      // zero length bucket.
      *centers = i ? *(centers - 1) : -std::numeric_limits<float>::infinity();
    } else {
      *centers = std::accumulate(start, finish, 0.0) / static_cast<float>(finish - start);
    }
  }
}

const char kSeparatelyQuantizeVersion = 2;

} // namespace

void SeparatelyQuantize::UpdateConfigFromBinary(const BinaryFormat &file, uint64_t offset, Config &config) {
  unsigned char buffer[3];
  file.ReadForConfig(buffer, 3, offset);
  char version = buffer[0];
  config.prob_bits = buffer[1];
  config.backoff_bits = buffer[2];
  if (version != kSeparatelyQuantizeVersion) UTIL_THROW(FormatLoadException, "This file has quantization version " << (unsigned)version << " but the code expects version " << (unsigned)kSeparatelyQuantizeVersion);
}

void SeparatelyQuantize::SetupMemory(void *base, unsigned char order, const Config &config) {
  prob_bits_ = config.prob_bits;
  backoff_bits_ = config.backoff_bits;
  // We need the reserved values.
  if (config.prob_bits == 0) UTIL_THROW(ConfigException, "You can't quantize probability to zero");
  if (config.backoff_bits == 0) UTIL_THROW(ConfigException, "You can't quantize backoff to zero");
  if (config.prob_bits > 25) UTIL_THROW(ConfigException, "For efficiency reasons, quantizing probability supports at most 25 bits.  Currently you have requested " << static_cast<unsigned>(config.prob_bits) << " bits.");
  if (config.backoff_bits > 25) UTIL_THROW(ConfigException, "For efficiency reasons, quantizing backoff supports at most 25 bits.  Currently you have requested " << static_cast<unsigned>(config.backoff_bits) << " bits.");
  // Reserve 8 byte header for bit counts.
  actual_base_ = static_cast<uint8_t*>(base);
  float *start = reinterpret_cast<float*>(actual_base_ + 8);
  for (unsigned char i = 0; i < order - 2; ++i) {
    tables_[i][0] = Bins(prob_bits_, start);
    start += (1ULL << prob_bits_);
    tables_[i][1] = Bins(backoff_bits_, start);
    start += (1ULL << backoff_bits_);
  }
  longest_ = tables_[order - 2][0] = Bins(prob_bits_, start);
}

void SeparatelyQuantize::Train(uint8_t order, std::vector<float> &prob, std::vector<float> &backoff) {
  TrainProb(order, prob);

  // Backoff
  float *centers = tables_[order - 2][1].Populate();
  *(centers++) = kNoExtensionBackoff;
  *(centers++) = kExtensionBackoff;
  MakeBins(backoff, centers, (1ULL << backoff_bits_) - 2);
}

void SeparatelyQuantize::TrainProb(uint8_t order, std::vector<float> &prob) {
  float *centers = tables_[order - 2][0].Populate();
  MakeBins(prob, centers, (1ULL << prob_bits_));
}

void SeparatelyQuantize::FinishedLoading(const Config &config) {
  uint8_t *actual_base = actual_base_;
  *(actual_base++) = kSeparatelyQuantizeVersion; // version
  *(actual_base++) = config.prob_bits;
  *(actual_base++) = config.backoff_bits;
}

} // namespace ngram
} // namespace lm
