#include "lm/interpolate/normalize.hh"

#include "lm/common/compare.hh"
#include "lm/common/ngram_stream.hh"
#include "lm/interpolate/backoff_matrix.hh"
#include "lm/interpolate/bounded_sequence_encoding.hh"
#include "lm/interpolate/interpolate_info.hh"
#include "lm/interpolate/merge_probabilities.hh"
#include "lm/weights.hh"
#include "lm/word_index.hh"
#include "util/fixed_array.hh"
#include "util/scoped.hh"
#include "util/stream/stream.hh"
#include "util/stream/rewindable_stream.hh"

#include <functional>
#include <queue>
#include <vector>

namespace lm { namespace interpolate {
namespace {

class BackoffQueueEntry {
  public:
    BackoffQueueEntry(float &entry, const util::stream::ChainPosition &position)
      : entry_(entry), stream_(position) {
      entry_ = 0.0;
    }

    operator bool() const { return stream_; }

    NGramHeader operator*() const { return *stream_; }
    const NGramHeader *operator->() const { return &*stream_; }

    void Enter() {
      entry_ = stream_->Value().backoff;
    }

    BackoffQueueEntry &Next() {
      entry_ = 0.0;
      ++stream_;
      return *this;
    }

  private:
    float &entry_;
    NGramStream<ProbBackoff> stream_;
};

struct PtrGreater : public std::binary_function<const BackoffQueueEntry *, const BackoffQueueEntry *, bool> {
  bool operator()(const BackoffQueueEntry *first, const BackoffQueueEntry *second) const {
    return SuffixLexicographicLess<NGramHeader>()(**second, **first);
  }
};

class EntryOwner : public util::FixedArray<BackoffQueueEntry> {
  public:
    void push_back(float &entry, const util::stream::ChainPosition &position) {
      new (end()) BackoffQueueEntry(entry, position);
      Constructed();
    }
};

std::size_t MaxOrder(const util::FixedArray<util::stream::ChainPositions> &model) {
  std::size_t ret = 0;
  for (const util::stream::ChainPositions *m = model.begin(); m != model.end(); ++m) {
    ret = std::max(ret, m->size());
  }
  return ret;
}

class BackoffManager {
  public:
    explicit BackoffManager(const util::FixedArray<util::stream::ChainPositions> &models)
      : entered_(MaxOrder(models)), matrix_(models.size(), MaxOrder(models)), skip_write_(MaxOrder(models)) {
      std::size_t total = 0;
      for (const util::stream::ChainPositions *m = models.begin(); m != models.end(); ++m) {
        total += m->size();
      }
      for (std::size_t i = 0; i < MaxOrder(models); ++i) {
        entered_.push_back(models.size());
      }
      owner_.Init(total);
      for (const util::stream::ChainPositions *m = models.begin(); m != models.end(); ++m) {
        for (const util::stream::ChainPosition *j = m->begin(); j != m->end(); ++j) {
          owner_.push_back(matrix_.Backoff(m - models.begin(), j - m->begin()), *j);
          if (owner_.back()) {
            queue_.push(&owner_.back());
          }
        }
      }
    }

    void SetupSkip(std::size_t order, util::stream::Stream &stream) {
      skip_write_[order - 2] = &stream;
    }

    // Move up the backoffs for the given n-gram.  The n-grams must be provided
    // in suffix lexicographic order.
    void Enter(const NGramHeader &to) {
      // Check that we exited properly.
      for (std::size_t i = to.Order() - 1; i < entered_.size(); ++i) {
        assert(entered_[i].empty());
      }
      SuffixLexicographicLess<NGramHeader> less;
      while (!queue_.empty() && less(**queue_.top(), to))
        SkipRecord();
      while (TopMatches(to)) {
        BackoffQueueEntry *matches = queue_.top();
        entered_[to.Order() - 1].push_back(matches);
        matches->Enter();
        queue_.pop();
      }
    }

    void Exit(std::size_t order_minus_1) {
      for (BackoffQueueEntry **i = entered_[order_minus_1].begin(); i != entered_[order_minus_1].end(); ++i) {
        if ((*i)->Next())
          queue_.push(*i);
      }
      entered_[order_minus_1].clear();
    }

    float Get(std::size_t model, std::size_t order_minus_1) const {
      return matrix_.Backoff(model, order_minus_1);
    }

    void Finish() {
      while (!queue_.empty())
        SkipRecord();
    }

  private:
    void SkipRecord() {
      BackoffQueueEntry *top = queue_.top();
      queue_.pop();
      // Is this the last instance of the n-gram?
      if (!TopMatches(**top)) {
        // An n-gram is being skipped.  Called once per skipped n-gram,
        // regardless of how many models it comes from.
        *reinterpret_cast<float*>(skip_write_[(*top)->Order() - 1]->Get()) = 0.0;
        ++*skip_write_[(*top)->Order() - 1];
      }
      if (top->Next())
        queue_.push(top);
    }

    bool TopMatches(const NGramHeader &header) const {
      return !queue_.empty() && (*queue_.top())->Order() == header.Order() && std::equal(header.begin(), header.end(), (*queue_.top())->begin());
    }

    EntryOwner owner_;
    std::priority_queue<BackoffQueueEntry*, std::vector<BackoffQueueEntry*>, PtrGreater> queue_;

    // Indexed by order then just all the matching models.
    util::FixedArray<util::FixedArray<BackoffQueueEntry*> > entered_;

    BackoffMatrix matrix_;

    std::vector<util::stream::Stream*> skip_write_;
};

typedef long double Accum;

// Handles n-grams of the same order, using recursion to call another instance
// for higher orders.
class Recurse {
  public:
    Recurse(
        const InterpolateInfo &info, // Must stay alive the entire time.
        std::size_t order,
        const util::stream::ChainPosition &merged_probs,
        const util::stream::ChainPosition &prob_out,
        const util::stream::ChainPosition &backoff_out,
        BackoffManager &backoffs,
        Recurse *higher) // higher is null for the highest order.
      : order_(order),
        encoding_(MakeEncoder(info, order)),
        input_(merged_probs, PartialProbGamma(order, encoding_.EncodedLength())),
        prob_out_(prob_out),
        backoff_out_(backoff_out),
        backoffs_(backoffs),
        lambdas_(&*info.lambdas.begin()),
        higher_(higher),
        decoded_backoffs_(info.Models()),
        extended_context_(order - 1) {
      // This is only for bigrams and above.  Summing unigrams is a much easier case.
      assert(order >= 2);
    }

    // context = w_1^{n-1}
    // z_lower = Z(w_2^{n-1})
    // Input:
    //   Merged probabilities without backoff applied in input_.
    //   Backoffs via backoffs_.
    // Calculates:
    //   Z(w_1^{n-1}): intermediate only.
    //   p_I(x | w_1^{n-1}) for all x: w_1^{n-1}x exists: Written to prob_out_.
    //   b_I(w_1^{n-1}): Written to backoff_out_.
    void SameContext(const NGramHeader &context, Accum z_lower) {
      assert(context.size() == order_ - 1);
      backoffs_.Enter(context);
      prob_out_.Mark();

      // This is the backoff term that applies when one assumes everything backs off:
      // \prod_i b_i(w_1^{n-1})^{\lambda_i}.
      Accum backoff_once = 0.0;
      for (std::size_t m = 0; m < decoded_backoffs_.size(); ++m) {
        backoff_once += lambdas_[m] * backoffs_.Get(m, order_ - 2);
      }

      Accum z_delta = 0.0;
      std::size_t count = 0;
      for (; input_ && std::equal(context.begin(), context.end(), input_->begin()); ++input_, ++prob_out_, ++count) {
        // Apply backoffs to probabilities.
        // TODO: change bounded sequence encoding to have an iterator for decoding instead of doing a copy here.
        encoding_.Decode(input_->FromBegin(), &*decoded_backoffs_.begin());
        for (std::size_t m = 0; m < NumModels(); ++m) {
          // Apply the backoffs as instructed for model m.
          float accumulated = 0.0;
          // Change backoffs for [order it backed off to, order - 1) except
          // with 0-indexing.  There is still the potential to charge backoff
          // for order - 1, which is done later.  The backoffs charged here
          // are b_m(w_{n-1}^{n-1}) ... b_m(w_2^{n-1})
          for (unsigned char backed_to = decoded_backoffs_[m]; backed_to < order_ - 2; ++backed_to) {
            accumulated += backoffs_.Get(m, backed_to);
          }
          float lambda = lambdas_[m];
          // Lower p(x | w_2^{n-1}) gets all the backoffs except the highest.
          input_->LowerProb() += accumulated * lambda;
          // Charge the backoff b(w_1^{n-1}) if applicable, but only to attain p(x | w_1^{n-1})
          if (decoded_backoffs_[m] < order_ - 1) {
            accumulated += backoffs_.Get(m, order_ - 2);
          }
          input_->Prob() += accumulated * lambda;
        }
        // TODO: better precision/less operations here.
        z_delta += pow(10.0, input_->Prob()) - pow(10.0, input_->LowerProb() + backoff_once);

        // Write unnormalized probability record.
        std::copy(input_->begin(), input_->end(), reinterpret_cast<WordIndex*>(prob_out_.Get()));
        ProbWrite() = input_->Prob();
      }
      // TODO numerical precision.
      Accum z = log10(pow(10.0, z_lower + backoff_once) + z_delta);

      // Normalize.
      prob_out_.Rewind();
      for (std::size_t i = 0; i < count; ++i, ++prob_out_) {
        ProbWrite() -= z;
      }
      // This allows the stream to release data.
      prob_out_.Mark();

      // Output backoff.
      *reinterpret_cast<float*>(backoff_out_.Get()) = z_lower + backoff_once - z;
      ++backoff_out_;

      if (higher_.get())
        higher_->ExtendContext(context, z);

      backoffs_.Exit(order_ - 2);
    }

    // Call is given a context and z(context).
    // Evaluates y context x for all y,x.
    void ExtendContext(const NGramHeader &middle, Accum z_lower) {
      assert(middle.size() == order_ - 2);
      // Copy because the input will advance.  TODO avoid this copy by sharing amongst classes.
      std::copy(middle.begin(), middle.end(), extended_context_.begin() + 1);
      while (input_ && std::equal(middle.begin(), middle.end(), input_->begin() + 1)) {
        *extended_context_.begin() = *input_->begin();
        SameContext(NGramHeader(&*extended_context_.begin(), order_ - 1), z_lower);
      }
    }

    void Finish() {
      assert(!input_);
      prob_out_.Poison();
      backoff_out_.Poison();
      if (higher_.get())
        higher_->Finish();
    }

    // The BackoffManager class also injects backoffs when it skips ahead e.g. b(</s>) = 1
    util::stream::Stream &BackoffStream() { return backoff_out_; }

  private:
    // Write the probability to the correct place in prob_out_.  Should use a proxy but currently incompatible with RewindableStream.
    float &ProbWrite() {
      return *reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(prob_out_.Get()) + order_ * sizeof(WordIndex));
    }

    std::size_t NumModels() const { return decoded_backoffs_.size(); }

    const std::size_t order_;

    const BoundedSequenceEncoding encoding_;

    ProxyStream<PartialProbGamma> input_;
    util::stream::RewindableStream prob_out_;
    util::stream::Stream backoff_out_;

    BackoffManager &backoffs_;
    const float *const lambdas_;

    // Higher order instance of this same class.
    util::scoped_ptr<Recurse> higher_;

    // Temporary in SameContext.
    std::vector<unsigned char> decoded_backoffs_;
    // Temporary in ExtendContext.
    std::vector<WordIndex> extended_context_;
};

class Thread {
  public:
    Thread(const InterpolateInfo &info, util::FixedArray<util::stream::ChainPositions> &models_by_order, util::stream::Chains &prob_out, util::stream::Chains &backoff_out)
      : info_(info), models_by_order_(models_by_order), prob_out_(prob_out), backoff_out_(backoff_out) {}

    void Run(const util::stream::ChainPositions &merged_probabilities) {
      // Unigrams do not have enocded backoff info.
      ProxyStream<PartialProbGamma> in(merged_probabilities[0], PartialProbGamma(1, 0));
      util::stream::RewindableStream prob_write(prob_out_[0]);
      Accum z = 0.0;
      prob_write.Mark();
      WordIndex count = 0;
      for (; in; ++in, ++prob_write, ++count) {
        // Note assumption that probabilitity comes first
        memcpy(prob_write.Get(), in.Get(), sizeof(WordIndex) + sizeof(float));
        z += pow(10.0, in->Prob());
      }
      // TODO HACK TODO: lmplz outputs p(<s>) = 1 to get q to compute nicely.  That will always result in 1.0 more than it should be.
      z -= 1.0;
      float log_z = log10(z);
      prob_write.Rewind();
      // Normalize unigram probabilities.
      for (WordIndex i = 0; i < count; ++i, ++prob_write) {
        *reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(prob_write.Get()) + sizeof(WordIndex)) -= log_z;
      }
      prob_write.Poison();

      // Now setup the higher orders.
      util::scoped_ptr<Recurse> higher_order;
      BackoffManager backoffs(models_by_order_);
      std::size_t max_order = merged_probabilities.size();
      for (std::size_t order = max_order; order >= 2; --order) {
        higher_order.reset(new Recurse(info_, order, merged_probabilities[order - 1], prob_out_[order - 1], backoff_out_[order - 2], backoffs, higher_order.release()));
        backoffs.SetupSkip(order, higher_order->BackoffStream());
      }
      if (max_order > 1) {
        higher_order->ExtendContext(NGramHeader(NULL, 0), log_z);
        backoffs.Finish();
        higher_order->Finish();
      }
    }

  private:
    const InterpolateInfo info_;
    util::FixedArray<util::stream::ChainPositions> &models_by_order_;
    util::stream::ChainPositions prob_out_;
    util::stream::ChainPositions backoff_out_;
};

} // namespace

void Normalize(const InterpolateInfo &info, util::FixedArray<util::stream::ChainPositions> &models_by_order, util::stream::Chains &merged_probabilities, util::stream::Chains &prob_out, util::stream::Chains &backoff_out) {
  assert(prob_out.size() == backoff_out.size() + 1);
  // Arbitrarily put the thread on the merged_probabilities Chains.
  merged_probabilities >> Thread(info, models_by_order, prob_out, backoff_out);
}

}} // namespaces
