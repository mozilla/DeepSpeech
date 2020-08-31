#include "ctc_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

#include "decoder_utils.h"
#include "ThreadPool.h"
#include "fst/fstlib.h"
#include "path_trie.h"


int
DecoderState::init(const Alphabet& alphabet,
                   size_t beam_size,
                   double cutoff_prob,
                   size_t cutoff_top_n,
                   std::shared_ptr<Scorer> ext_scorer)
{
  // assign special ids
  abs_time_step_ = 0;
  space_id_ = alphabet.GetSpaceLabel();
  blank_id_ = alphabet.GetSize();

  beam_size_ = beam_size;
  cutoff_prob_ = cutoff_prob;
  cutoff_top_n_ = cutoff_top_n;
  ext_scorer_ = ext_scorer;
  start_expanding_ = false;

  // init prefixes' root
  PathTrie *root = new PathTrie;
  root->score = root->log_prob_b_prev = 0.0;
  prefix_root_.reset(root);
  prefixes_.push_back(root);

  if (ext_scorer && (bool)(ext_scorer_->dictionary)) {
    // no need for std::make_shared<>() since Copy() does 'new' behind the doors
    auto dict_ptr = std::shared_ptr<PathTrie::FstType>(ext_scorer->dictionary->Copy(true));
    root->set_dictionary(dict_ptr);
    auto matcher = std::make_shared<fst::SortedMatcher<PathTrie::FstType>>(*dict_ptr, fst::MATCH_INPUT);
    root->set_matcher(matcher);
  }

  return 0;
}

void
DecoderState::next(const double *probs,
                   int time_dim,
                   int class_dim)
{
  // prefix search over time
  for (size_t rel_time_step = 0; rel_time_step < time_dim; ++rel_time_step, ++abs_time_step_) {
    auto *prob = &probs[rel_time_step*class_dim];

    // At the start of the decoding process, we delay beam expansion so that
    // timings on the first letters is not incorrect. As soon as we see a
    // timestep with blank probability lower than 0.999, we start expanding
    // beams.
    if (prob[blank_id_] < 0.999) {
      start_expanding_ = true;
    }

    // If not expanding yet, just continue to next timestep.
    if (!start_expanding_) {
      continue;
    }

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    if (ext_scorer_) {
      size_t num_prefixes = std::min(prefixes_.size(), beam_size_);
      std::partial_sort(prefixes_.begin(),
                        prefixes_.begin() + num_prefixes,
                        prefixes_.end(),
                        prefix_compare);

      min_cutoff = prefixes_[num_prefixes - 1]->score +
                   std::log(prob[blank_id_]) - std::max(0.0, ext_scorer_->beta);
      full_beam = (num_prefixes == beam_size_);
    }

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(prob, class_dim, cutoff_prob_, cutoff_top_n_);
    // loop over class dim
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < prefixes_.size() && i < beam_size_; ++i) {
        auto prefix = prefixes_[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        assert(prefix->is_empty() || !prefix->timesteps.empty());

        // blank
        if (c == blank_id_) {
          // compute probability of current path
          float log_p = log_prob_c + prefix->score;

          // combine current path with previous ones with the same prefix
          // the blank label comes last, so we can compare log_prob_nb_cur with log_p
          if (prefix->log_prob_nb_cur < log_p) {
              prefix->timesteps_cur = prefix->timesteps;
          }
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_p);
          continue;
        }

        // repeated character
        if (c == prefix->character) {
          // compute probability of current path
          float log_p = log_prob_c + prefix->log_prob_nb_prev;

          // combine current path with previous ones with the same prefix
          if (prefix->log_prob_nb_cur < log_p) {
              prefix->timesteps_cur = prefix->timesteps;
          }
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_p);
        }

        // get new prefix
        auto prefix_new = prefix->get_path_trie(c, log_prob_c);

        if (prefix_new != nullptr) {
          // compute timesteps of current path
          std::vector<unsigned int> timesteps_new=prefix->timesteps;
          timesteps_new.push_back(abs_time_step_);

          // compute probability of current path
          float log_p = -NUM_FLT_INF;

          if (c == prefix->character &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (c != prefix->character) {
            log_p = log_prob_c + prefix->score;
          }

          if (ext_scorer_) {
            // skip scoring the space in word based LMs
            PathTrie* prefix_to_score;
            if (ext_scorer_->is_utf8_mode()) {
              prefix_to_score = prefix_new;
            } else {
              prefix_to_score = prefix;
            }

            // language model scoring
            if (ext_scorer_->is_scoring_boundary(prefix_to_score, c)) {
              float score = 0.0;
              std::vector<std::string> ngram;
              ngram = ext_scorer_->make_ngram(prefix_to_score);
              bool bos = ngram.size() < ext_scorer_->get_max_order();
              score = ext_scorer_->get_log_cond_prob(ngram, bos) * ext_scorer_->alpha;
              log_p += score;
              log_p += ext_scorer_->beta;
            }
          }

          // combine current path with previous ones with the same prefix
          if (prefix_new->log_prob_nb_cur < log_p) {
              prefix_new->timesteps_cur = timesteps_new;
          }
          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over alphabet

    // update log probs
    prefixes_.clear();
    prefix_root_->iterate_to_vec(prefixes_);

    // only preserve top beam_size prefixes
    if (prefixes_.size() > beam_size_) {
      std::nth_element(prefixes_.begin(),
                       prefixes_.begin() + beam_size_,
                       prefixes_.end(),
                       prefix_compare);
      for (size_t i = beam_size_; i < prefixes_.size(); ++i) {
        prefixes_[i]->remove();
      }

      // Remove the elements from std::vector
      prefixes_.resize(beam_size_);
    }
  }  // end of loop over time
}

std::vector<Output>
DecoderState::decode(size_t num_results) const
{
  std::vector<PathTrie*> prefixes_copy = prefixes_;
  std::unordered_map<const PathTrie*, float> scores;
  for (PathTrie* prefix : prefixes_copy) {
    scores[prefix] = prefix->score;
  }

  // score the last word of each prefix that doesn't end with space
  if (ext_scorer_) {
    for (size_t i = 0; i < beam_size_ && i < prefixes_copy.size(); ++i) {
      PathTrie* prefix = prefixes_copy[i];
      PathTrie* prefix_boundary = ext_scorer_->is_utf8_mode() ? prefix : prefix->parent;
      if (prefix_boundary && !ext_scorer_->is_scoring_boundary(prefix_boundary, prefix->character)) {
        float score = 0.0;
        std::vector<std::string> ngram = ext_scorer_->make_ngram(prefix);
        bool bos = ngram.size() < ext_scorer_->get_max_order();
        score = ext_scorer_->get_log_cond_prob(ngram, bos) * ext_scorer_->alpha;
        score += ext_scorer_->beta;
        scores[prefix] += score;
      }
    }
  }

  using namespace std::placeholders;
  size_t num_returned = std::min(prefixes_copy.size(), num_results);
  std::partial_sort(prefixes_copy.begin(),
                    prefixes_copy.begin() + num_returned,
                    prefixes_copy.end(),
                    std::bind(prefix_compare_external, _1, _2, scores));

  std::vector<Output> outputs;
  outputs.reserve(num_returned);

  for (PathTrie* prefix : prefixes_copy) {
    Output output;
    prefix->get_path_vec(output.tokens);
    output.timesteps  = prefix->timesteps;
    output.confidence = scores[prefix];
    outputs.push_back(output);
    if(outputs.size()>=num_returned) break;
  }

  return outputs;
}

std::vector<Output> ctc_beam_search_decoder(
    const double *probs,
    int time_dim,
    int class_dim,
    const Alphabet &alphabet,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    std::shared_ptr<Scorer> ext_scorer,
    size_t num_results)
{
  VALID_CHECK_EQ(alphabet.GetSize()+1, class_dim, "Number of output classes in acoustic model does not match number of labels in the alphabet file. Alphabet file must be the same one that was used to train the acoustic model.");
  DecoderState state;
  state.init(alphabet, beam_size, cutoff_prob, cutoff_top_n, ext_scorer);
  state.next(probs, time_dim, class_dim);
  return state.decode(num_results);
}

std::vector<std::vector<Output>>
ctc_beam_search_decoder_batch(
    const double *probs,
    int batch_size,
    int time_dim,
    int class_dim,
    const int* seq_lengths,
    int seq_lengths_size,
    const Alphabet &alphabet,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    std::shared_ptr<Scorer> ext_scorer,
    size_t num_results)
{
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  VALID_CHECK_EQ(batch_size, seq_lengths_size, "must have one sequence length per batch element");
  // thread pool
  ThreadPool pool(num_processes);

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<Output>>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder,
                                  &probs[i*time_dim*class_dim],
                                  seq_lengths[i],
                                  class_dim,
                                  alphabet,
                                  beam_size,
                                  cutoff_prob,
                                  cutoff_top_n,
                                  ext_scorer,
                                  num_results));
  }

  // get decoding results
  std::vector<std::vector<Output>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}
