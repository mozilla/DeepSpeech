#ifndef LM_CONFIG_H
#define LM_CONFIG_H

#include "lm/lm_exception.hh"
#include "util/mmap.hh"

#include <iosfwd>
#include <string>
#include <vector>

/* Configuration for ngram model.  Separate header to reduce pollution. */

namespace lm {

class EnumerateVocab;

namespace ngram {

struct Config {
  // EFFECTIVE FOR BOTH ARPA AND BINARY READS

  // (default true) print progress bar to messages
  bool show_progress;

  // Where to log messages including the progress bar.  Set to NULL for
  // silence.
  std::ostream *messages;

  std::ostream *ProgressMessages() const {
    return show_progress ? messages : 0;
  }

  // This will be called with every string in the vocabulary by the
  // constructor; it need only exist for the lifetime of the constructor.
  // See enumerate_vocab.hh for more detail.  Config does not take ownership;
  // just delete/let it go out of scope after the constructor exits.
  EnumerateVocab *enumerate_vocab;


  // ONLY EFFECTIVE WHEN READING ARPA

  // What to do when <unk> isn't in the provided model.
  WarningAction unknown_missing;
  // What to do when <s> or </s> is missing from the model.
  // If THROW_UP, the exception will be of type util::SpecialWordMissingException.
  WarningAction sentence_marker_missing;

  // What to do with a positive log probability.  For COMPLAIN and SILENT, map
  // to 0.
  WarningAction positive_log_probability;

  // The probability to substitute for <unk> if it's missing from the model.
  // No effect if the model has <unk> or unknown_missing == THROW_UP.
  float unknown_missing_logprob;

  // Size multiplier for probing hash table.  Must be > 1.  Space is linear in
  // this.  Time is probing_multiplier / (probing_multiplier - 1).  No effect
  // for sorted variant.
  // If you find yourself setting this to a low number, consider using the
  // TrieModel which has lower memory consumption.
  float probing_multiplier;

  // Amount of memory to use for building.  The actual memory usage will be
  // higher since this just sets sort buffer size.  Only applies to trie
  // models.
  std::size_t building_memory;

  // Template for temporary directory appropriate for passing to mkdtemp.
  // The characters XXXXXX are appended before passing to mkdtemp.  Only
  // applies to trie.  If empty, defaults to write_mmap.  If that's NULL,
  // defaults to input file name.
  std::string temporary_directory_prefix;

  // Level of complaining to do when loading from ARPA instead of binary format.
  enum ARPALoadComplain {ALL, EXPENSIVE, NONE};
  ARPALoadComplain arpa_complain;

  // While loading an ARPA file, also write out this binary format file.  Set
  // to NULL to disable.
  const char *write_mmap;

  enum WriteMethod {
    WRITE_MMAP, // Map the file directly.
    WRITE_AFTER // Write after we're done.
  };
  WriteMethod write_method;

  // Include the vocab in the binary file?  Only effective if write_mmap != NULL.
  bool include_vocab;


  // Left rest options.  Only used when the model includes rest costs.
  enum RestFunction {
    REST_MAX,   // Maximum of any score to the left
    REST_LOWER, // Use lower-order files given below.
  };
  RestFunction rest_function;
  // Only used for REST_LOWER.
  std::vector<std::string> rest_lower_files;


  // Quantization options.  Only effective for QuantTrieModel.  One value is
  // reserved for each of prob and backoff, so 2^bits - 1 buckets will be used
  // to quantize (and one of the remaining backoffs will be 0).
  uint8_t prob_bits, backoff_bits;

  // Bhiksha compression (simple form).  Only works with trie.
  uint8_t pointer_bhiksha_bits;


  // ONLY EFFECTIVE WHEN READING BINARY

  // How to get the giant array into memory: lazy mmap, populate, read etc.
  // See util/mmap.hh for details of MapMethod.
  util::LoadMethod load_method;


  // Set defaults.
  Config();
};

} /* namespace ngram */ } /* namespace lm */

#endif // LM_CONFIG_H
