#include "decoder_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

std::vector<std::pair<size_t, float>> get_pruned_log_probs(
    const double *prob_step,
    size_t class_dim,
    double cutoff_prob,
    size_t cutoff_top_n) {
  std::vector<std::pair<int, double>> prob_idx;
  for (size_t i = 0; i < class_dim; ++i) {
    prob_idx.push_back(std::pair<int, double>(i, prob_step[i]));
  }
  // pruning of vacobulary
  size_t cutoff_len = class_dim;
  if (cutoff_prob < 1.0 || cutoff_top_n < cutoff_len) {
    std::sort(
        prob_idx.begin(), prob_idx.end(), pair_comp_second_rev<int, double>);
    if (cutoff_prob < 1.0) {
      double cum_prob = 0.0;
      cutoff_len = 0;
      for (size_t i = 0; i < prob_idx.size(); ++i) {
        cum_prob += prob_idx[i].second;
        cutoff_len += 1;
        if (cum_prob >= cutoff_prob || cutoff_len >= cutoff_top_n) break;
      }
    }
    prob_idx = std::vector<std::pair<int, double>>(
        prob_idx.begin(), prob_idx.begin() + cutoff_len);
  }
  std::vector<std::pair<size_t, float>> log_prob_idx;
  for (size_t i = 0; i < cutoff_len; ++i) {
    log_prob_idx.push_back(std::pair<int, float>(
        prob_idx[i].first, log(prob_idx[i].second + NUM_FLT_MIN)));
  }
  return log_prob_idx;
}


std::vector<Output> get_beam_search_result(
    const std::vector<PathTrie *> &prefixes,
    size_t top_paths) {
  std::vector<Output> output_vecs;
  for (size_t i = 0; i < top_paths && i < prefixes.size(); ++i) {
    Output output;
    prefixes[i]->get_path_vec(output.tokens, output.timesteps);
    output.probability = -prefixes[i]->approx_ctc;
    output_vecs.push_back(output);
  }

  return output_vecs;
}

size_t get_utf8_str_len(const std::string &str) {
  size_t str_len = 0;
  for (char c : str) {
    str_len += ((c & 0xc0) != 0x80);
  }
  return str_len;
}

std::vector<std::string> split_utf8_str(const std::string &str) {
  std::vector<std::string> result;
  std::string out_str;

  for (char c : str) {
    if ((c & 0xc0) != 0x80)  // new UTF-8 character
    {
      if (!out_str.empty()) {
        result.push_back(out_str);
        out_str.clear();
      }
    }

    out_str.append(1, c);
  }
  result.push_back(out_str);
  return result;
}

std::vector<std::string> split_str(const std::string &s,
                                   const std::string &delim) {
  std::vector<std::string> result;
  std::size_t start = 0, delim_len = delim.size();
  while (true) {
    std::size_t end = s.find(delim, start);
    if (end == std::string::npos) {
      if (start < s.size()) {
        result.push_back(s.substr(start));
      }
      break;
    }
    if (end > start) {
      result.push_back(s.substr(start, end - start));
    }
    start = end + delim_len;
  }
  return result;
}

bool prefix_compare(const PathTrie *x, const PathTrie *y) {
  if (x->score == y->score) {
    if (x->character == y->character) {
      return false;
    } else {
      return (x->character < y->character);
    }
  } else {
    return x->score > y->score;
  }
}

bool prefix_compare_external(const PathTrie *x, const PathTrie *y, const std::unordered_map<const PathTrie*, float>& scores) {
  if (scores.at(x) == scores.at(y)) {
    if (x->character == y->character) {
      return false;
    } else {
      return (x->character < y->character);
    }
  } else {
    return scores.at(x) > scores.at(y);
  }
}

void add_word_to_fst(const std::vector<int> &word,
                     fst::StdVectorFst *dictionary) {
  if (dictionary->NumStates() == 0) {
    fst::StdVectorFst::StateId start = dictionary->AddState();
    assert(start == 0);
    dictionary->SetStart(start);
  }
  fst::StdVectorFst::StateId src = dictionary->Start();
  fst::StdVectorFst::StateId dst;
  for (auto c : word) {
    dst = dictionary->AddState();
    dictionary->AddArc(src, fst::StdArc(c, c, 0, dst));
    src = dst;
  }
  dictionary->SetFinal(dst, fst::StdArc::Weight::One());
}

bool add_word_to_dictionary(
    const std::string &word,
    const std::unordered_map<std::string, int> &char_map,
    bool add_space,
    int SPACE_ID,
    fst::StdVectorFst *dictionary) {
  auto characters = split_utf8_str(word);

  std::vector<int> int_word;

  for (auto &c : characters) {
    if (c == " ") {
      int_word.push_back(SPACE_ID);
    } else {
      auto int_c = char_map.find(c);
      if (int_c != char_map.end()) {
        int_word.push_back(int_c->second);
      } else {
        return false;  // return without adding
      }
    }
  }

  if (add_space) {
    int_word.push_back(SPACE_ID);
  }

  add_word_to_fst(int_word, dictionary);
  return true;  // return with successful adding
}