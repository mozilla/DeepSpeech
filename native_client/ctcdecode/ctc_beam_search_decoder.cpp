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

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;

std::vector<Output> ctc_beam_search_decoder(
    const double *probs,
    int time_dim,
    int class_dim,
    const Alphabet &alphabet,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer) {
  // dimension check
  VALID_CHECK_EQ(class_dim, alphabet.GetSize()+1,
                 "The shape of probs does not match with "
                 "the shape of the vocabulary");

  // assign special ids
  int space_id = alphabet.GetSpaceLabel();
  int blank_id = alphabet.GetSize();

  // init prefixes' root
  PathTrie root;
  root.score = root.log_prob_b_prev = 0.0;
  std::vector<PathTrie *> prefixes;
  prefixes.push_back(&root);

  if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
    auto dict_ptr = ext_scorer->dictionary->Copy(true);
    root.set_dictionary(dict_ptr);
    auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
    root.set_matcher(matcher);
  }

  // prefix search over time
  for (size_t time_step = 0; time_step < time_dim; ++time_step) {
    auto *prob = &probs[time_step*class_dim];

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    if (ext_scorer != nullptr) {
      size_t num_prefixes = std::min(prefixes.size(), beam_size);
      std::sort(
          prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);
      min_cutoff = prefixes[num_prefixes - 1]->score +
                   std::log(prob[blank_id]) - std::max(0.0, ext_scorer->beta);
      full_beam = (num_prefixes == beam_size);
    }

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(prob, class_dim, cutoff_prob, cutoff_top_n);
    // loop over chars
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
        auto prefix = prefixes[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        // blank
        if (c == blank_id) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }
        // repeated character
        if (c == prefix->character) {
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }
        // get new prefix
        auto prefix_new = prefix->get_path_trie(c, time_step);

        if (prefix_new != nullptr) {
          float log_p = -NUM_FLT_INF;

          if (c == prefix->character &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (c != prefix->character) {
            log_p = log_prob_c + prefix->score;
          }

          // language model scoring
          if (ext_scorer != nullptr &&
              (c == space_id || ext_scorer->is_character_based())) {
            PathTrie *prefix_to_score = nullptr;
            // skip scoring the space
            if (ext_scorer->is_character_based()) {
              prefix_to_score = prefix_new;
            } else {
              prefix_to_score = prefix;
            }

            float score = 0.0;
            std::vector<std::string> ngram;
            ngram = ext_scorer->make_ngram(prefix_to_score);
            score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
            log_p += score;
            log_p += ext_scorer->beta;
          }
          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over vocabulary


    prefixes.clear();
    // update log probs
    root.iterate_to_vec(prefixes);

    // only preserve top beam_size prefixes
    if (prefixes.size() >= beam_size) {
      std::nth_element(prefixes.begin(),
                       prefixes.begin() + beam_size,
                       prefixes.end(),
                       prefix_compare);
      for (size_t i = beam_size; i < prefixes.size(); ++i) {
        prefixes[i]->remove();
      }
    }
  }  // end of loop over time

  // score the last word of each prefix that doesn't end with space
  if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
    for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
      auto prefix = prefixes[i];
      if (!prefix->is_empty() && prefix->character != space_id) {
        float score = 0.0;
        std::vector<std::string> ngram = ext_scorer->make_ngram(prefix);
        score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
        score += ext_scorer->beta;
        prefix->score += score;
      }
    }
  }

  size_t num_prefixes = std::min(prefixes.size(), beam_size);
  std::sort(prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);

  // compute aproximate ctc score as the return score, without affecting the
  // return order of decoding result. To delete when decoder gets stable.
  for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
    double approx_ctc = prefixes[i]->score;
    if (ext_scorer != nullptr) {
      std::vector<int> output;
      std::vector<int> timesteps;
      prefixes[i]->get_path_vec(output, timesteps);
      auto prefix_length = output.size();
      auto words = ext_scorer->split_labels(output);
      // remove word insert
      approx_ctc = approx_ctc - prefix_length * ext_scorer->beta;
      // remove language model weight:
      approx_ctc -= (ext_scorer->get_sent_log_prob(words)) * ext_scorer->alpha;
    }
    prefixes[i]->approx_ctc = approx_ctc;
  }

  return get_beam_search_result(prefixes, beam_size);
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
    Scorer *ext_scorer) {
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
                                  ext_scorer));
  }

  // get decoding results
  std::vector<std::vector<Output>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}
