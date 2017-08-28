#ifndef LM_QUANTIZE_H
#define LM_QUANTIZE_H

#include "lm/blank.hh"
#include "lm/config.hh"
#include "lm/max_order.hh"
#include "lm/model_type.hh"
#include "util/bit_packing.hh"

#include <algorithm>
#include <vector>

#include <stdint.h>

#include <iostream>

namespace lm {
namespace ngram {

struct Config;
class BinaryFormat;

/* Store values directly and don't quantize. */
class DontQuantize {
  public:
    static const ModelType kModelTypeAdd = static_cast<ModelType>(0);
    static void UpdateConfigFromBinary(const BinaryFormat &, uint64_t, Config &) {}
    static uint64_t Size(uint8_t /*order*/, const Config &/*config*/) { return 0; }
    static uint8_t MiddleBits(const Config &/*config*/) { return 63; }
    static uint8_t LongestBits(const Config &/*config*/) { return 31; }

    class MiddlePointer {
      public:
        MiddlePointer(const DontQuantize & /*quant*/, unsigned char /*order_minus_2*/, util::BitAddress address) : address_(address) {}

        MiddlePointer() : address_(NULL, 0) {}

        bool Found() const {
          return address_.base != NULL;
        }

        float Prob() const {
          return util::ReadNonPositiveFloat31(address_.base, address_.offset);
        }

        float Backoff() const {
          return util::ReadFloat32(address_.base, address_.offset + 31);
        }

        float Rest() const { return Prob(); }

        void Write(float prob, float backoff) {
          util::WriteNonPositiveFloat31(address_.base, address_.offset, prob);
          util::WriteFloat32(address_.base, address_.offset + 31, backoff);
        }

      private:
        util::BitAddress address_;
    };

    class LongestPointer {
      public:
        explicit LongestPointer(const DontQuantize &/*quant*/, util::BitAddress address) : address_(address) {}

        LongestPointer() : address_(NULL, 0) {}

        bool Found() const {
          return address_.base != NULL;
        }

        float Prob() const {
          return util::ReadNonPositiveFloat31(address_.base, address_.offset);
        }

        void Write(float prob) {
          util::WriteNonPositiveFloat31(address_.base, address_.offset, prob);
        }

      private:
        util::BitAddress address_;
    };

    DontQuantize() {}

    void SetupMemory(void * /*start*/, unsigned char /*order*/, const Config & /*config*/) {}

    static const bool kTrain = false;
    // These should never be called because kTrain is false.
    void Train(uint8_t /*order*/, std::vector<float> &/*prob*/, std::vector<float> &/*backoff*/) {}
    void TrainProb(uint8_t, std::vector<float> &/*prob*/) {}

    void FinishedLoading(const Config &) {}
};

class SeparatelyQuantize {
  private:
    class Bins {
      public:
        // Sigh C++ default constructor
        Bins() {}

        Bins(uint8_t bits, float *begin) : begin_(begin), end_(begin_ + (1ULL << bits)), bits_(bits), mask_((1ULL << bits) - 1) {}

        float *Populate() { return begin_; }

        uint64_t EncodeProb(float value) const {
          return Encode(value, 0);
        }

        uint64_t EncodeBackoff(float value) const {
          if (value == 0.0) {
            return HasExtension(value) ? kExtensionQuant : kNoExtensionQuant;
          }
          return Encode(value, 2);
        }

        float Decode(std::size_t off) const { return begin_[off]; }

        uint8_t Bits() const { return bits_; }

        uint64_t Mask() const { return mask_; }

      private:
        uint64_t Encode(float value, size_t reserved) const {
          const float *above = std::lower_bound(static_cast<const float*>(begin_) + reserved, end_, value);
          if (above == begin_ + reserved) return reserved;
          if (above == end_) return end_ - begin_ - 1;
          return above - begin_ - (value - *(above - 1) < *above - value);
        }

        float *begin_;
        const float *end_;
        uint8_t bits_;
        uint64_t mask_;
    };

  public:
    static const ModelType kModelTypeAdd = kQuantAdd;

    static void UpdateConfigFromBinary(const BinaryFormat &file, uint64_t offset, Config &config);

    static uint64_t Size(uint8_t order, const Config &config) {
      uint64_t longest_table = (static_cast<uint64_t>(1) << static_cast<uint64_t>(config.prob_bits)) * sizeof(float);
      uint64_t middle_table = (static_cast<uint64_t>(1) << static_cast<uint64_t>(config.backoff_bits)) * sizeof(float) + longest_table;
      // unigrams are currently not quantized so no need for a table.
      return (order - 2) * middle_table + longest_table + /* for the bit counts and alignment padding) */ 8;
    }

    static uint8_t MiddleBits(const Config &config) { return config.prob_bits + config.backoff_bits; }
    static uint8_t LongestBits(const Config &config) { return config.prob_bits; }

    class MiddlePointer {
      public:
        MiddlePointer(const SeparatelyQuantize &quant, unsigned char order_minus_2, const util::BitAddress &address) : bins_(quant.GetTables(order_minus_2)), address_(address) {}

        MiddlePointer() : address_(NULL, 0) {}

        bool Found() const { return address_.base != NULL; }

        float Prob() const {
          return ProbBins().Decode(util::ReadInt25(address_.base, address_.offset + BackoffBins().Bits(), ProbBins().Bits(), ProbBins().Mask()));
        }

        float Backoff() const {
          return BackoffBins().Decode(util::ReadInt25(address_.base, address_.offset, BackoffBins().Bits(), BackoffBins().Mask()));
        }

        float Rest() const { return Prob(); }

        void Write(float prob, float backoff) const {
          util::WriteInt57(address_.base, address_.offset, ProbBins().Bits() + BackoffBins().Bits(),
              (ProbBins().EncodeProb(prob) << BackoffBins().Bits()) | BackoffBins().EncodeBackoff(backoff));
        }

      private:
        const Bins &ProbBins() const { return bins_[0]; }
        const Bins &BackoffBins() const { return bins_[1]; }
        const Bins *bins_;

        util::BitAddress address_;
    };

    class LongestPointer {
      public:
        LongestPointer(const SeparatelyQuantize &quant, const util::BitAddress &address) : table_(&quant.LongestTable()), address_(address) {}

        LongestPointer() : address_(NULL, 0) {}

        bool Found() const { return address_.base != NULL; }

        void Write(float prob) const {
          util::WriteInt25(address_.base, address_.offset, table_->Bits(), table_->EncodeProb(prob));
        }

        float Prob() const {
          return table_->Decode(util::ReadInt25(address_.base, address_.offset, table_->Bits(), table_->Mask()));
        }

      private:
        const Bins *table_;
        util::BitAddress address_;
    };

    SeparatelyQuantize() {}

    void SetupMemory(void *start, unsigned char order, const Config &config);

    static const bool kTrain = true;
    // Assumes 0.0 is removed from backoff.
    void Train(uint8_t order, std::vector<float> &prob, std::vector<float> &backoff);
    // Train just probabilities (for longest order).
    void TrainProb(uint8_t order, std::vector<float> &prob);

    void FinishedLoading(const Config &config);

    const Bins *GetTables(unsigned char order_minus_2) const { return tables_[order_minus_2]; }

    const Bins &LongestTable() const { return longest_; }

  private:
    Bins tables_[KENLM_MAX_ORDER - 1][2];

    Bins longest_;

    uint8_t *actual_base_;

    uint8_t prob_bits_, backoff_bits_;
};

} // namespace ngram
} // namespace lm

#endif // LM_QUANTIZE_H
