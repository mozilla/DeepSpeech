// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Common classes for Multi Pushdown Transducer (MPDT) expansion/traversal.

#ifndef FST_EXTENSIONS_MPDT_MPDT_H_
#define FST_EXTENSIONS_MPDT_MPDT_H_

#include <array>
#include <functional>
#include <map>
#include <vector>

#include <fst/compat.h>
#include <fst/extensions/pdt/pdt.h>

namespace fst {

enum MPdtType {
  MPDT_READ_RESTRICT,   // Can only read from first empty stack
  MPDT_WRITE_RESTRICT,  // Can only write to first empty stack
  MPDT_NO_RESTRICT,     // No read-write restrictions
};

namespace internal {

// PLEASE READ THIS CAREFULLY:
//
// When USEVECTOR is set, the stack configurations --- the statewise
// representation of the StackId's for each substack --- is stored in a vector.
// I would like to do this using an array for efficiency reasons, thus the
// definition of StackConfig below. However, while this *works* in that tests
// pass, etc. It causes a memory leak in the compose and expand tests, evidently
// in the map[] that is being used to store the mapping between these
// StackConfigs and the external StackId that the caller sees. There are no
// memory leaks when I use a vector, only with this StackId. Why there should be
// memory leaks given that I am not mallocing anything is a mystery. In case you
// were wondering, clearing the map at the end does not help.

template <typename StackId, typename Level, Level nlevels>
struct StackConfig {
  StackConfig() : array_() {}

  StackConfig(const StackConfig<StackId, Level, nlevels> &config) {
    array_ = config.array_;
  }

  StackId &operator[](const int index) { return array_[index]; }

  const StackId &operator[](const int index) const { return array_[index]; }

  StackConfig &operator=(const StackConfig<StackId, Level, nlevels> &config) {
    if (this == &config) return *this;
    array_ = config.array_;
    return *this;
  }

  std::array<StackId, nlevels> array_;
};

template <typename StackId, typename Level, Level nlevels>
class CompConfig {
 public:
  using Config = StackConfig<StackId, Level, nlevels>;

  bool operator()(const Config &x, const Config &y) const {
    for (Level level = 0; level < nlevels; ++level) {
      if (x.array_[level] < y.array_[level]) {
        return true;
      } else if (x.array_[level] > y.array_[level]) {
        return false;
      }
    }
    return false;
  }
};

// Defines the KeyPair type used as the key to MPdtStack.paren_id_map_. The hash
// function is provided as a separate struct to match templating syntax.
template <typename Level>
struct KeyPair {
  Level level;
  size_t underlying_id;

  KeyPair(Level level, size_t id) : level(level), underlying_id(id) {}

  inline bool operator==(const KeyPair<Level> &rhs) const {
    return level == rhs.level && underlying_id == rhs.underlying_id;
  }
};

template <typename Level>
struct KeyPairHasher {
  inline size_t operator()(const KeyPair<Level> &keypair) const {
    return std::hash<Level>()(keypair.level) ^
           (std::hash<size_t>()(keypair.underlying_id) << 1);
  }
};

template <typename StackId, typename Level, Level nlevels = 2,
          MPdtType restrict = MPDT_READ_RESTRICT>
class MPdtStack {
 public:
  using Label = Level;
  using Config = StackConfig<StackId, Level, nlevels>;
  using ConfigToStackId =
      std::map<Config, StackId, CompConfig<StackId, Level, nlevels>>;

  MPdtStack(const std::vector<std::pair<Label, Label>> &parens,
            const std::vector<Level> &assignments);

  MPdtStack(const MPdtStack &mstack);

  ~MPdtStack() {
    for (Level level = 0; level < nlevels; ++level) delete stacks_[level];
  }

  StackId Find(StackId stack_id, Label label);

  // For now we do not implement Pop since this is needed only for
  // ShortestPath().

  // For Top we find the first non-empty config, and find the paren ID of that
  // (or -1) if there is none, then map that to the external stack_id to return.
  std::ptrdiff_t Top(StackId stack_id) const {
    if (stack_id == -1) return -1;
    const auto config = InternalStackIds(stack_id);
    Level level = 0;
    StackId underlying_id = -1;
    for (; level < nlevels; ++level) {
      if (!Empty(config, level)) {
        underlying_id = stacks_[level]->Top(config[level]);
        break;
      }
    }
    if (underlying_id == -1) return -1;
    const auto it = paren_id_map_.find(KeyPair<Level>(level, underlying_id));
    if (it == paren_id_map_.end()) return -1;  // NB: shouldn't happen.
    return it->second;
  }

  std::ptrdiff_t ParenId(Label label) const {
    const auto it = paren_map_.find(label);
    return it != paren_map_.end() ? it->second : -1;
  }

  // TODO(rws): For debugging purposes only: remove later.
  string PrintConfig(const Config &config) const {
    string result = "[";
    for (Level i = 0; i < nlevels; ++i) {
      char s[128];
      snprintf(s, sizeof(s), "%d", config[i]);
      result += string(s);
      if (i < nlevels - 1) result += ", ";
    }
    result += "]";
    return result;
  }

  bool Error() { return error_; }

  // Each component stack has an internal stack ID for a given configuration and
  // label.
  // This function maps a configuration of those to the stack ID the caller
  // sees.
  inline StackId ExternalStackId(const Config &config) {
    const auto it = config_to_stack_id_map_.find(config);
    StackId result;
    if (it == config_to_stack_id_map_.end()) {
      result = next_stack_id_++;
      config_to_stack_id_map_.insert(
          std::pair<Config, StackId>(config, result));
      stack_id_to_config_map_[result] = config;
    } else {
      result = it->second;
    }
    return result;
  }

  // Retrieves the internal stack ID for a corresponding external stack ID.
  inline const Config InternalStackIds(StackId stack_id) const {
    auto it = stack_id_to_config_map_.find(stack_id);
    if (it == stack_id_to_config_map_.end()) {
      it = stack_id_to_config_map_.find(-1);
    }
    return it->second;
  }

  inline bool Empty(const Config &config, Level level) const {
    return config[level] <= 0;
  }

  inline bool AllEmpty(const Config &config) {
    for (Level level = 0; level < nlevels; ++level) {
      if (!Empty(config, level)) return false;
    }
    return true;
  }

  bool error_;
  Label min_paren_;
  Label max_paren_;
  // Stores level of each paren.
  std::unordered_map<Label, Label> paren_levels_;
  std::vector<std::pair<Label, Label>> parens_;  // As in pdt.h.
  std::unordered_map<Label, size_t> paren_map_;  // As in pdt.h.
  // Maps between internal paren_id and external paren_id.
  std::unordered_map<KeyPair<Level>, size_t, KeyPairHasher<Level>>
      paren_id_map_;
  // Maps between internal stack ids and external stack id.
  ConfigToStackId config_to_stack_id_map_;
  std::unordered_map<StackId, Config> stack_id_to_config_map_;
  StackId next_stack_id_;
  // Array of stacks.
  PdtStack<StackId, Label> *stacks_[nlevels];
};

template <typename StackId, typename Level, Level nlevels, MPdtType restrict>
MPdtStack<StackId, Level, nlevels, restrict>::MPdtStack(
    const std::vector<std::pair<Level, Level>> &parens,  // NB: Label = Level.
    const std::vector<Level> &assignments)
    : error_(false),
      min_paren_(kNoLabel),
      max_paren_(kNoLabel),
      parens_(parens),
      next_stack_id_(1) {
  using Label = Level;
  if (parens.size() != assignments.size()) {
    FSTERROR() << "MPdtStack: Parens of different size from assignments";
    error_ = true;
    return;
  }
  std::vector<std::pair<Label, Label>> vectors[nlevels];
  for (Level i = 0; i < assignments.size(); ++i) {
    // Assignments here start at 0, so assuming the human-readable version has
    // them starting at 1, we should subtract 1 here
    const auto level = assignments[i] - 1;
    if (level < 0 || level >= nlevels) {
      FSTERROR() << "MPdtStack: Specified level " << level << " out of bounds";
      error_ = true;
      return;
    }
    const auto &pair = parens[i];
    vectors[level].push_back(pair);
    paren_levels_[pair.first] = level;
    paren_levels_[pair.second] = level;
    paren_map_[pair.first] = i;
    paren_map_[pair.second] = i;
    const KeyPair<Level> key(level, vectors[level].size() - 1);
    paren_id_map_[key] = i;
    if (min_paren_ == kNoLabel || pair.first < min_paren_) {
      min_paren_ = pair.first;
    }
    if (pair.second < min_paren_) min_paren_ = pair.second;
    if (max_paren_ == kNoLabel || pair.first > max_paren_) {
      max_paren_ = pair.first;
    }
    if (pair.second > max_paren_) max_paren_ = pair.second;
  }
  using Config = StackConfig<StackId, Level, nlevels>;
  Config neg_one;
  Config zero;
  for (Level level = 0; level < nlevels; ++level) {
    stacks_[level] = new PdtStack<StackId, Label>(vectors[level]);
    neg_one[level] = -1;
    zero[level] = 0;
  }
  config_to_stack_id_map_[neg_one] = -1;
  config_to_stack_id_map_[zero] = 0;
  stack_id_to_config_map_[-1] = neg_one;
  stack_id_to_config_map_[0] = zero;
}

template <typename StackId, typename Level, Level nlevels, MPdtType restrict>
MPdtStack<StackId, Level, nlevels, restrict>::MPdtStack(
    const MPdtStack<StackId, Level, nlevels, restrict> &mstack)
    : error_(mstack.error_),
      min_paren_(mstack.min_paren_),
      max_paren_(mstack.max_paren_),
      parens_(mstack.parens_),
      next_stack_id_(mstack.next_stack_id_) {
  for (const auto &kv : mstack.paren_levels_) {
    paren_levels_[kv.first] = kv.second;
  }
  for (const auto &paren : mstack.parens_) parens_.push_back(paren);
  for (const auto &kv : mstack.paren_map_) {
    paren_map_[kv.first] = kv.second;
  }
  for (const auto &kv : mstack.paren_id_map_) {
    paren_id_map_[kv.first] = kv.second;
  }
  for (auto it = mstack.config_to_stack_id_map_.begin();
       it != mstack.config_to_stack_id_map_.end(); ++it) {
    config_to_stack_id_map_[it->first] = it->second;
  }
  for (const auto &kv : mstack.stack_id_to_config_map_) {
    using Config = StackConfig<StackId, Level, nlevels>;
    const Config config(kv.second);
    stack_id_to_config_map_[kv.first] = config;
  }
  for (Level level = 0; level < nlevels; ++level)
    stacks_[level] = mstack.stacks_[level];
}

template <typename StackId, typename Level, Level nlevels, MPdtType restrict>
StackId MPdtStack<StackId, Level, nlevels, restrict>::Find(StackId stack_id,
                                                           Level label) {
  // Non-paren.
  if (min_paren_ == kNoLabel || label < min_paren_ || label > max_paren_) {
    return stack_id;
  }
  const auto it = paren_map_.find(label);
  // Non-paren.
  if (it == paren_map_.end()) return stack_id;
  std::ptrdiff_t paren_id = it->second;
  // Gets the configuration associated with this stack_id.
  const auto config = InternalStackIds(stack_id);
  // Gets the level.
  const auto level = paren_levels_.find(label)->second;
  // If the label is an open paren we push:
  //
  // 1) if the restrict type is not MPDT_WRITE_RESTRICT, or
  // 2) the restrict type is MPDT_WRITE_RESTRICT, and all the stacks above the
  // level are empty.
  if (label == parens_[paren_id].first) {  // Open paren.
    if (restrict == MPDT_WRITE_RESTRICT) {
      for (Level upper_level = 0; upper_level < level; ++upper_level) {
        if (!Empty(config, upper_level)) return -1;  // Non-empty stack blocks.
      }
    }
    // If the label is an close paren we pop:
    //
    // 1) if the restrict type is not MPDT_READ_RESTRICT, or
    // 2) the restrict type is MPDT_READ_RESTRICT, and all the stacks above the
    // level are empty.
  } else if (restrict == MPDT_READ_RESTRICT) {
    for (Level lower_level = 0; lower_level < level; ++lower_level) {
      if (!Empty(config, lower_level)) return -1;  // Non-empty stack blocks.
    }
  }
  const auto nid = stacks_[level]->Find(config[level], label);
  // If the new ID is -1, that means that there is no valid transition at the
  // level we want.
  if (nid == -1) {
    return -1;
  } else {
    using Config = StackConfig<StackId, Level, nlevels>;
    Config nconfig(config);
    nconfig[level] = nid;
    return ExternalStackId(nconfig);
  }
}

}  // namespace internal
}  // namespace fst

#endif  // FST_EXTENSIONS_MPDT_MPDT_H_
