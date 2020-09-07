#include "path_trie.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "decoder_utils.h"

PathTrie::PathTrie() {
  log_prob_b_prev = -NUM_FLT_INF;
  log_prob_nb_prev = -NUM_FLT_INF;
  log_prob_b_cur = -NUM_FLT_INF;
  log_prob_nb_cur = -NUM_FLT_INF;
  log_prob_c = -NUM_FLT_INF;
  score = -NUM_FLT_INF;

  ROOT_ = -1;
  character = ROOT_;
  exists_ = true;
  parent = nullptr;

  dictionary_ = nullptr;
  dictionary_state_ = 0;
  has_dictionary_ = false;

  matcher_ = nullptr;
}

PathTrie::~PathTrie() {
  for (auto child : children_) {
    delete child.second;
  }
}

PathTrie* PathTrie::get_path_trie(unsigned int new_char, float cur_log_prob_c, bool reset) {
  auto child = children_.begin();
  for (; child != children_.end(); ++child) {
    if (child->first == new_char) {
      break;
    }
  }
  if (child != children_.end()) {
    if (!child->second->exists_) {
      child->second->exists_ = true;
      child->second->log_prob_b_prev = -NUM_FLT_INF;
      child->second->log_prob_nb_prev = -NUM_FLT_INF;
      child->second->log_prob_b_cur = -NUM_FLT_INF;
      child->second->log_prob_nb_cur = -NUM_FLT_INF;
    }
    return child->second;
  } else {
    if (has_dictionary_) {
      matcher_->SetState(dictionary_state_);
      bool found = matcher_->Find(new_char + 1);
      if (!found) {
        // Adding this character causes word outside dictionary
        auto FSTZERO = fst::TropicalWeight::Zero();
        auto final_weight = dictionary_->Final(dictionary_state_);
        bool is_final = (final_weight != FSTZERO);
        if (is_final && reset) {
          dictionary_state_ = dictionary_->Start();
        }
        return nullptr;
      } else {
        PathTrie* new_path = new PathTrie;
        new_path->character = new_char;
        new_path->parent = this;
        new_path->dictionary_ = dictionary_;
        new_path->has_dictionary_ = true;
        new_path->matcher_ = matcher_;
        new_path->log_prob_c = cur_log_prob_c;

        // set spell checker state
        // check to see if next state is final
        auto FSTZERO = fst::TropicalWeight::Zero();
        auto final_weight = dictionary_->Final(matcher_->Value().nextstate);
        bool is_final = (final_weight != FSTZERO);
        if (is_final && reset) {
          // restart spell checker at the start state
          new_path->dictionary_state_ = dictionary_->Start();
        } else {
          // go to next state
          new_path->dictionary_state_ = matcher_->Value().nextstate;
        }

        children_.push_back(std::make_pair(new_char, new_path));
        return new_path;
      }
    } else {
      PathTrie* new_path = new PathTrie;
      new_path->character = new_char;
      new_path->parent = this;
      new_path->log_prob_c = cur_log_prob_c;
      children_.push_back(std::make_pair(new_char, new_path));
      return new_path;
    }
  }
}

void PathTrie::get_path_vec(std::vector<unsigned int>& output) {
  // Recursive call: recurse back until stop condition, then append data in
  // correct order as we walk back down the stack in the lines below.
  if (parent != nullptr) {
    parent->get_path_vec(output);
  }
  if (character != ROOT_) {
    output.push_back(character);
  }
}

PathTrie* PathTrie::get_prev_grapheme(std::vector<unsigned int>& output,
                                      const Alphabet& alphabet)
{
  PathTrie* stop = this;
  if (character == ROOT_) {
    return stop;
  }
  // Recursive call: recurse back until stop condition, then append data in
  // correct order as we walk back down the stack in the lines below.
  if (!byte_is_codepoint_boundary(alphabet.DecodeSingle(character)[0])) {
    stop = parent->get_prev_grapheme(output, alphabet);
  }
  output.push_back(character);
  return stop;
}

int PathTrie::distance_to_codepoint_boundary(unsigned char *first_byte,
                                             const Alphabet& alphabet)
{
  if (byte_is_codepoint_boundary(alphabet.DecodeSingle(character)[0])) {
    *first_byte = (unsigned char)character + 1;
    return 1;
  }
  if (parent != nullptr && parent->character != ROOT_) {
    return 1 + parent->distance_to_codepoint_boundary(first_byte, alphabet);
  }
  assert(false); // unreachable
  return 0;
}

PathTrie* PathTrie::get_prev_word(std::vector<unsigned int>& output,
                                  const Alphabet& alphabet)
{
  PathTrie* stop = this;
  if (character == alphabet.GetSpaceLabel() || character == ROOT_) {
    return stop;
  }
  // Recursive call: recurse back until stop condition, then append data in
  // correct order as we walk back down the stack in the lines below.
  if (parent != nullptr) {
    stop = parent->get_prev_word(output, alphabet);
  }
  output.push_back(character);
  return stop;
}

void PathTrie::iterate_to_vec(std::vector<PathTrie*>& output) {
  // previous_timesteps might point to ancestors' timesteps
  // therefore, children must be uptaded first
  for (auto child : children_) {
    child.second->iterate_to_vec(output);
  }
  if (exists_) {
    log_prob_b_prev = log_prob_b_cur;
    log_prob_nb_prev = log_prob_nb_cur;

    log_prob_b_cur = -NUM_FLT_INF;
    log_prob_nb_cur = -NUM_FLT_INF;

    score = log_sum_exp(log_prob_b_prev, log_prob_nb_prev);

    if (previous_timesteps != nullptr) {
      timesteps = nullptr;
      for (auto const& child : previous_timesteps->children) {
        if (child->data == new_timestep) {
            timesteps=child;
            break;
        }
      }
      if (timesteps == nullptr){
          timesteps = add_child(previous_timesteps, new_timestep);
      }
    }
    previous_timesteps=nullptr;

    output.push_back(this);
  }
}

void PathTrie::remove() {
  exists_ = false;

  if (children_.size() == 0) {
    for (auto child = parent->children_.begin(); child != parent->children_.end(); ++child) {
      if (child->first == character) {
        parent->children_.erase(child);
        break;
      }
    }

    if (parent->children_.size() == 0 && !parent->exists_) {
      parent->remove();
    }

    delete this;
  }
}

void PathTrie::set_dictionary(std::shared_ptr<PathTrie::FstType> dictionary) {
  dictionary_ = dictionary;
  dictionary_state_ = dictionary_->Start();
  has_dictionary_ = true;
}

void PathTrie::set_matcher(std::shared_ptr<fst::SortedMatcher<FstType>> matcher) {
  matcher_ = matcher;
}

#ifdef DEBUG
void PathTrie::vec(std::vector<PathTrie*>& out) {
  if (parent != nullptr) {
    parent->vec(out);
  }
  out.push_back(this);
}

void PathTrie::print(const Alphabet& a) {
  std::vector<PathTrie*> chain;
  vec(chain);
  std::string tr;
  printf("characters:\t ");
  for (PathTrie* el : chain) {
    printf("%X ", (unsigned char)(el->character));
    if (el->character != ROOT_) {
      tr.append(a.DecodeSingle(el->character));
    }
  }
  printf("\ntimesteps:\t ");
  for (unsigned int timestep : get_history(timesteps)) {
    printf("%d ", timestep);
  }
  printf("\n");
  printf("transcript:\t %s\n", tr.c_str());
}
#endif // DEBUG
