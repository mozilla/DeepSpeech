/* Efficient left and right language model state for sentence fragments.
 * Intended usage:
 * Store ChartState with every chart entry.
 * To do a rule application:
 * 1. Make a ChartState object for your new entry.
 * 2. Construct RuleScore.
 * 3. Going from left to right, call Terminal or NonTerminal.
 *   For terminals, just pass the vocab id.
 *   For non-terminals, pass that non-terminal's ChartState.
 *     If your decoder expects scores inclusive of subtree scores (i.e. you
 *     label entries with the highest-scoring path), pass the non-terminal's
 *     score as prob.
 *     If your decoder expects relative scores and will walk the chart later,
 *     pass prob = 0.0.
 *     In other words, the only effect of prob is that it gets added to the
 *     returned log probability.
 * 4. Call Finish.  It returns the log probability.
 *
 * There's a couple more details:
 * Do not pass <s> to Terminal as it is formally not a word in the sentence,
 * only context.  Instead, call BeginSentence.  If called, it should be the
 * first call after RuleScore is constructed (since <s> is always the
 * leftmost).
 *
 * If the leftmost RHS is a non-terminal, it's faster to call BeginNonTerminal.
 *
 * Hashing and sorting comparison operators are provided.   All state objects
 * are POD.  If you intend to use memcmp on raw state objects, you must call
 * ZeroRemaining first, as the value of array entries beyond length is
 * otherwise undefined.
 *
 * Usage is of course not limited to chart decoding.  Anything that generates
 * sentence fragments missing left context could benefit.  For example, a
 * phrase-based decoder could pre-score phrases, storing ChartState with each
 * phrase, even if hypotheses are generated left-to-right.
 */

#ifndef LM_LEFT_H
#define LM_LEFT_H

#include "lm/max_order.hh"
#include "lm/state.hh"
#include "lm/return.hh"

#include "util/murmur_hash.hh"

#include <algorithm>

namespace lm {
namespace ngram {

template <class M> class RuleScore {
  public:
    explicit RuleScore(const M &model, ChartState &out) : model_(model), out_(&out), left_done_(false), prob_(0.0) {
      out.left.length = 0;
      out.right.length = 0;
    }

    void BeginSentence() {
      out_->right = model_.BeginSentenceState();
      // out_->left is empty.
      left_done_ = true;
    }

    void Terminal(WordIndex word) {
      State copy(out_->right);
      FullScoreReturn ret(model_.FullScore(copy, word, out_->right));
      if (left_done_) { prob_ += ret.prob; return; }
      if (ret.independent_left) {
        prob_ += ret.prob;
        left_done_ = true;
        return;
      }
      out_->left.pointers[out_->left.length++] = ret.extend_left;
      prob_ += ret.rest;
      if (out_->right.length != copy.length + 1)
        left_done_ = true;
    }

    // Faster version of NonTerminal for the case where the rule begins with a non-terminal.
    void BeginNonTerminal(const ChartState &in, float prob = 0.0) {
      prob_ = prob;
      *out_ = in;
      left_done_ = in.left.full;
    }

    void NonTerminal(const ChartState &in, float prob = 0.0) {
      prob_ += prob;

      if (!in.left.length) {
        if (in.left.full) {
          for (const float *i = out_->right.backoff; i < out_->right.backoff + out_->right.length; ++i) prob_ += *i;
          left_done_ = true;
          out_->right = in.right;
        }
        return;
      }

      if (!out_->right.length) {
        out_->right = in.right;
        if (left_done_) {
          prob_ += model_.UnRest(in.left.pointers, in.left.pointers + in.left.length, 1);
          return;
        }
        if (out_->left.length) {
          left_done_ = true;
        } else {
          out_->left = in.left;
          left_done_ = in.left.full;
        }
        return;
      }

      float backoffs[KENLM_MAX_ORDER - 1], backoffs2[KENLM_MAX_ORDER - 1];
      float *back = backoffs, *back2 = backoffs2;
      unsigned char next_use = out_->right.length;

      // First word
      if (ExtendLeft(in, next_use, 1, out_->right.backoff, back)) return;

      // Words after the first, so extending a bigram to begin with
      for (unsigned char extend_length = 2; extend_length <= in.left.length; ++extend_length) {
        if (ExtendLeft(in, next_use, extend_length, back, back2)) return;
        std::swap(back, back2);
      }

      if (in.left.full) {
        for (const float *i = back; i != back + next_use; ++i) prob_ += *i;
        left_done_ = true;
        out_->right = in.right;
        return;
      }

      // Right state was minimized, so it's already independent of the new words to the left.
      if (in.right.length < in.left.length) {
        out_->right = in.right;
        return;
      }

      // Shift exisiting words down.
      for (WordIndex *i = out_->right.words + next_use - 1; i >= out_->right.words; --i) {
        *(i + in.right.length) = *i;
      }
      // Add words from in.right.
      std::copy(in.right.words, in.right.words + in.right.length, out_->right.words);
      // Assemble backoff composed on the existing state's backoff followed by the new state's backoff.
      std::copy(in.right.backoff, in.right.backoff + in.right.length, out_->right.backoff);
      std::copy(back, back + next_use, out_->right.backoff + in.right.length);
      out_->right.length = in.right.length + next_use;
    }

    float Finish() {
      // A N-1-gram might extend left and right but we should still set full to true because it's an N-1-gram.
      out_->left.full = left_done_ || (out_->left.length == model_.Order() - 1);
      return prob_;
    }

    void Reset() {
      prob_ = 0.0;
      left_done_ = false;
      out_->left.length = 0;
      out_->right.length = 0;
    }
    void Reset(ChartState &replacement) {
      out_ = &replacement;
      Reset();
    }

  private:
    bool ExtendLeft(const ChartState &in, unsigned char &next_use, unsigned char extend_length, const float *back_in, float *back_out) {
      ProcessRet(model_.ExtendLeft(
            out_->right.words, out_->right.words + next_use, // Words to extend into
            back_in, // Backoffs to use
            in.left.pointers[extend_length - 1], extend_length, // Words to be extended
            back_out, // Backoffs for the next score
            next_use)); // Length of n-gram to use in next scoring.
      if (next_use != out_->right.length) {
        left_done_ = true;
        if (!next_use) {
          // Early exit.
          out_->right = in.right;
          prob_ += model_.UnRest(in.left.pointers + extend_length, in.left.pointers + in.left.length, extend_length + 1);
          return true;
        }
      }
      // Continue scoring.
      return false;
    }

    void ProcessRet(const FullScoreReturn &ret) {
      if (left_done_) {
        prob_ += ret.prob;
        return;
      }
      if (ret.independent_left) {
        prob_ += ret.prob;
        left_done_ = true;
        return;
      }
      out_->left.pointers[out_->left.length++] = ret.extend_left;
      prob_ += ret.rest;
    }

    const M &model_;

    ChartState *out_;

    bool left_done_;

    float prob_;
};

} // namespace ngram
} // namespace lm

#endif // LM_LEFT_H
