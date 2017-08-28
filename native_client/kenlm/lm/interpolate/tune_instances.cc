/* Load tuning instances and filter underlying models to them.  A tuning
 * instance is an n-gram in the tuning file.  To tune towards these, we want
 * the correct probability p_i(w_n | w_1^{n-1}) from each model as well as
 * all the denominators p_i(v | w_1^{n-1}) that appear in normalization.
 *
 * In other words, we filter the models to only those n-grams whose context
 * appears in the tuning data.  This can be divided into two categories:
 * - All unigrams.  This goes into Instances::ln_unigrams_
 * - Bigrams and above whose context appears in the tuning data.  These are
 *   known as extensions.  We only care about the longest extension for each
 *   w_1^{n-1}v since that is what will be used for the probability.
 * Because there is a large number of extensions (we tried keeping them in RAM
 * and ran out), the streaming framework is used to keep track of extensions
 * and sort them so they can be streamed in.  Downstream code
 * (tune_derivatives.hh) takes a stream of extensions ordered by tuning
 * instance, the word v, and the model the extension came from.
 */
#include "lm/interpolate/tune_instances.hh"

#include "lm/common/compare.hh"
#include "lm/common/joint_order.hh"
#include "lm/common/model_buffer.hh"
#include "lm/common/ngram_stream.hh"
#include "lm/common/renumber.hh"
#include "lm/enumerate_vocab.hh"
#include "lm/interpolate/merge_vocab.hh"
#include "lm/interpolate/universal_vocab.hh"
#include "lm/lm_exception.hh"
#include "util/file_piece.hh"
#include "util/murmur_hash.hh"
#include "util/stream/chain.hh"
#include "util/stream/io.hh"
#include "util/stream/sort.hh"
#include "util/tokenize_piece.hh"

#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace lm { namespace interpolate {

// gcc 4.6 complains about uninitialized when sort code is generated for a 4-byte POD.  But that sort code is never used.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
bool Extension::operator<(const Extension &other) const {
  if (instance != other.instance)
    return instance < other.instance;
  if (word != other.word)
    return word < other.word;
  if (model != other.model)
    return model < other.model;
  return false;
}
#pragma GCC diagnostic pop

namespace {

// An extension without backoff weights applied yet.
#pragma pack(push)
#pragma pack(1)
struct InitialExtension {
  Extension ext;
  // Order from which it came.
  uint8_t order;
};
#pragma pack(pop)

struct InitialExtensionCompare {
  bool operator()(const void *first, const void *second) const {
    return reinterpret_cast<const InitialExtension *>(first)->ext < reinterpret_cast<const InitialExtension *>(second)->ext;
  }
};

// Intended use
// For each model:
//   stream through orders jointly in suffix order:
//     Call MatchedBackoff for full matches.
//     Call Exit when the context matches.
//   Call FinishModel with the unigram probability of the correct word, get full
//   probability in return.
// Use backoffs_out to adjust records that were written to the stream.
// backoffs_out(model, order - 1) is the penalty for matching order.
class InstanceMatch {
  public:
    InstanceMatch(Matrix &backoffs_out, const WordIndex correct)
      : seen_(std::numeric_limits<WordIndex>::max()),
        backoffs_(backoffs_out),
        correct_(correct), correct_from_(1), correct_ln_prob_(std::numeric_limits<float>::quiet_NaN()) {}

    void MatchedBackoff(ModelIndex model, uint8_t order, float ln_backoff) {
      backoffs_(model, order - 1) = ln_backoff;
    }

    // We only want the highest-order matches, which are the first to be exited for a given word.
    void Exit(const InitialExtension &from, util::stream::Stream &out) {
      if (from.ext.word == seen_) return;
      seen_ = from.ext.word;
      *static_cast<InitialExtension*>(out.Get()) = from;
      ++out;
      if (UTIL_UNLIKELY(correct_ == from.ext.word)) {
        correct_from_ = from.order;
        correct_ln_prob_ = from.ext.ln_prob;
      }
    }

    WordIndex Correct() const { return correct_; }

    // Call this after each model has been passed through.  Provide the unigram
    // probability of the correct word (which follows the given context).
    // This function will return the fully-backed-off probability of the correct
    // word.
    float FinishModel(ModelIndex model, float correct_ln_unigram) {
      seen_ = std::numeric_limits<WordIndex>::max();
      // Turn backoffs into multiplied values (added in log space).
      // So backoffs_(model, order - 1) is the penalty for matching order.
      float accum = 0.0;
      for (int order = backoffs_.cols() - 1; order >= 0; --order) {
        accum += backoffs_(model, order);
        backoffs_(model, order) = accum;
      }
      if (correct_from_ == 1) {
        correct_ln_prob_ = correct_ln_unigram;
      }
      if (correct_from_ - 1 < backoffs_.cols()) {
        correct_ln_prob_ += backoffs_(model, correct_from_ - 1);
      }
      correct_from_ = 1;
      return correct_ln_prob_;
    }

  private:
    // What's the last word we've seen?  Used to act only on exiting the longest match.
    WordIndex seen_;

    Matrix &backoffs_;

    const WordIndex correct_;

    // These only apply to the most recent model.
    uint8_t correct_from_;

    float correct_ln_prob_;
};

// Forward information to multiple instances of a context.  So if the tuning
// set contains
//   a b c d e
//   a b c d e
// there's one DispatchContext for a b c d which calls two InstanceMatch, one
// for each tuning instance.  This might be to inform them about a b c d g in
// one of the models.
class DispatchContext {
  public:
    void Register(InstanceMatch &context) {
      registered_.push_back(&context);
    }

    void MatchedBackoff(ModelIndex model, uint8_t order, float ln_backoff) {
      for (std::vector<InstanceMatch*>::iterator i = registered_.begin(); i != registered_.end(); ++i)
        (*i)->MatchedBackoff(model, order, ln_backoff);
    }

    void Exit(InitialExtension &from, util::stream::Stream &out, const InstanceMatch *base_instance) {
      for (std::vector<InstanceMatch*>::iterator i = registered_.begin(); i != registered_.end(); ++i) {
        from.ext.instance = *i - base_instance;
        (*i)->Exit(from, out);
      }
    }

  private:
    // TODO make these offsets in a big array rather than separately allocated.
    std::vector<InstanceMatch*> registered_;
};

// Map from n-gram hash to contexts in the tuning data.  TODO: probing hash table?
typedef boost::unordered_map<uint64_t, DispatchContext> ContextMap;

// Handle all the orders of a single model at once.
class JointOrderCallback {
  public:
    JointOrderCallback(
        std::size_t model,
        std::size_t full_order_minus_1,
        ContextMap &contexts,
        util::stream::Stream &out,
        const InstanceMatch *base_instance)
      : full_order_minus_1_(full_order_minus_1),
        contexts_(contexts),
        out_(out),
        base_instance_(base_instance) {
      ext_.ext.model = model;
    }

    void Enter(std::size_t order_minus_1, const void *data) {}

    void Exit(std::size_t order_minus_1, void *data) {
      // Match the full n-gram for backoffs.
      if (order_minus_1 != full_order_minus_1_) {
        NGram<ProbBackoff> gram(data, order_minus_1 + 1);
        ContextMap::iterator i = contexts_.find(util::MurmurHashNative(gram.begin(), gram.Order() * sizeof(WordIndex)));
        if (UTIL_UNLIKELY(i != contexts_.end())) {
          i->second.MatchedBackoff(ext_.ext.model, gram.Order(), gram.Value().backoff * M_LN10);
        }
      }
      // Match the context of the n-gram to indicate it's an extension.
      ContextMap::iterator i = contexts_.find(util::MurmurHashNative(data, order_minus_1 * sizeof(WordIndex)));
      if (UTIL_UNLIKELY(i != contexts_.end())) {
        NGram<Prob> gram(data, order_minus_1 + 1);
        // model is already set.
        // instance is set by DispatchContext.
        // That leaves word, ln_prob, and order.
        ext_.ext.word = *(gram.end() - 1);
        ext_.ext.ln_prob = gram.Value().prob * M_LN10;
        ext_.order = order_minus_1 + 1;
        // model was already set in the constructor.
        // ext_.ext.instance is set by the Exit call.
        i->second.Exit(ext_, out_, base_instance_);
      }
    }

    void Run(const util::stream::ChainPositions &positions) {
      JointOrder<JointOrderCallback, SuffixOrder>(positions, *this);
    }

  private:
    const std::size_t full_order_minus_1_;

    // Mapping is constant but values are being manipulated to tell them about
    // n-grams.
    ContextMap &contexts_;

    // Reused variable.  model is set correctly.
    InitialExtension ext_;

    util::stream::Stream &out_;

    const InstanceMatch *const base_instance_;
};

// This populates the ln_unigrams_ matrix.  It can (and should for efficiency)
// be run in the same scan as JointOrderCallback.
class ReadUnigrams {
  public:
    explicit ReadUnigrams(Matrix::ColXpr out) : out_(out) {}

    // Read renumbered unigrams, fill with <unk> otherwise.
    void Run(const util::stream::ChainPosition &position) {
      NGramStream<ProbBackoff> stream(position);
      assert(stream);
      Accum unk = stream->Value().prob * M_LN10;
      WordIndex previous = 0;
      for (; stream; ++stream) {
        WordIndex word = *stream->begin();
        out_.segment(previous, word - previous) = Vector::Constant(word - previous, unk);
        out_(word) = stream->Value().prob * M_LN10;
        //backoffs are used by JointOrderCallback.
        previous = word + 1;
      }
      out_.segment(previous, out_.rows() - previous) = Vector::Constant(out_.rows() - previous, unk);
    }

  private:
    Matrix::ColXpr out_;
};

// Read tuning data into an array of vocab ids.  The vocab ids are agreed with MergeVocab.
class IdentifyTuning : public EnumerateVocab {
  public:
    IdentifyTuning(int tuning_file, std::vector<WordIndex> &out) : indices_(out) {
      indices_.clear();
      StringPiece line;
      std::size_t counter = 0;
      std::vector<std::size_t> &eos = words_[util::MurmurHashNative("</s>", 4)];
      for (util::FilePiece f(tuning_file); f.ReadLineOrEOF(line);) {
        for (util::TokenIter<util::BoolCharacter, true> word(line, util::kSpaces); word; ++word) {
          UTIL_THROW_IF(*word == "<s>" || *word == "</s>", FormatLoadException, "Illegal word in tuning data: " << *word);
          words_[util::MurmurHashNative(word->data(), word->size())].push_back(counter++);
        }
        eos.push_back(counter++);
      }
      // Also get <s>
      indices_.resize(counter + 1);
      words_[util::MurmurHashNative("<s>", 3)].push_back(indices_.size() - 1);
    }

    // Apply ids as they come out of MergeVocab if they match.
    void Add(WordIndex id, const StringPiece &str) {
      boost::unordered_map<uint64_t, std::vector<std::size_t> >::iterator i = words_.find(util::MurmurHashNative(str.data(), str.size()));
      if (i != words_.end()) {
        for (std::vector<std::size_t>::iterator j = i->second.begin(); j != i->second.end(); ++j) {
          indices_[*j] = id;
        }
      }
    }

    WordIndex FinishGetBOS() {
      WordIndex ret = indices_.back();
      indices_.pop_back();
      return ret;
    }

  private:
    // array of words in tuning data.
    std::vector<WordIndex> &indices_;

    // map from hash(string) to offsets in indices_.
    boost::unordered_map<uint64_t, std::vector<std::size_t> > words_;
};

} // namespace

// Store information about the first iteration.
class ExtensionsFirstIteration {
  public:
    explicit ExtensionsFirstIteration(std::size_t instances, std::size_t models, std::size_t max_order, util::stream::Chain &extension_input, const util::stream::SortConfig &config)
      : backoffs_by_instance_(new std::vector<Matrix>(instances)), sort_(extension_input, config) {
      // Initialize all the backoff matrices to zeros.
      for (std::vector<Matrix>::iterator i = backoffs_by_instance_->begin(); i != backoffs_by_instance_->end(); ++i) {
        *i = Matrix::Zero(models, max_order);
      }
    }

    Matrix &WriteBackoffs(std::size_t instance) {
      return (*backoffs_by_instance_)[instance];
    }

    // Get the backoff all the way to unigram for a particular tuning instance and model.
    Accum FullBackoff(std::size_t instance, std::size_t model) const {
      return (*backoffs_by_instance_)[instance](model, 0);
    }

    void Merge(std::size_t lazy_memory) {
      sort_.Merge(lazy_memory);
      lazy_memory_ = lazy_memory;
    }

    void Output(util::stream::Chain &chain) {
      sort_.Output(chain, lazy_memory_);
      chain >> ApplyBackoffs(backoffs_by_instance_);
    }

  private:
    class ApplyBackoffs {
      public:
        explicit ApplyBackoffs(boost::shared_ptr<std::vector<Matrix> > backoffs_by_instance)
          : backoffs_by_instance_(backoffs_by_instance) {}

        void Run(const util::stream::ChainPosition &position) {
          // There should always be tuning instances.
          const std::vector<Matrix> &backoffs = *backoffs_by_instance_;
          assert(!backoffs.empty());
          uint8_t max_order = backoffs.front().cols();
          for (util::stream::Stream stream(position); stream; ++stream) {
            InitialExtension &ini = *reinterpret_cast<InitialExtension*>(stream.Get());
            assert(ini.order > 1); // If it's an extension, it should be higher than a unigram.
            if (ini.order != max_order) {
              ini.ext.ln_prob += backoffs[ini.ext.instance](ini.ext.model, ini.order - 1);
            }
          }
        }

      private:
        boost::shared_ptr<std::vector<Matrix> > backoffs_by_instance_;
    };

    // Array of complete backoff matrices by instance.
    // Each matrix is by model, then by order.
    // Would have liked to use a tensor but it's not that well supported.
    // This is a shared pointer so that ApplyBackoffs can run after this class is gone.
    boost::shared_ptr<std::vector<Matrix> > backoffs_by_instance_;

    // This sorts and stores all the InitialExtensions.
    util::stream::Sort<InitialExtensionCompare> sort_;

    std::size_t lazy_memory_;
};

Instances::Instances(int tune_file, const std::vector<StringPiece> &model_names, const InstancesConfig &config) : temp_prefix_(config.sort.temp_prefix) {
  // All the memory from stack variables here should go away before merge sort of the instances.
  {
    util::FixedArray<ModelBuffer> models(model_names.size());

    // Load tuning set and join vocabulary.
    std::vector<WordIndex> vocab_sizes;
    vocab_sizes.reserve(model_names.size());
    util::FixedArray<int> vocab_files(model_names.size());
    std::size_t max_order = 0;
    for (std::vector<StringPiece>::const_iterator i = model_names.begin(); i != model_names.end(); ++i) {
      models.push_back(*i);
      vocab_sizes.push_back(models.back().Counts()[0]);
      vocab_files.push_back(models.back().VocabFile());
      max_order = std::max(max_order, models.back().Order());
    }
    UniversalVocab vocab(vocab_sizes);
    std::vector<WordIndex> tuning_words;
    WordIndex combined_vocab_size;
    {
      IdentifyTuning identify(tune_file, tuning_words);
      combined_vocab_size = MergeVocab(vocab_files, vocab, identify);
      bos_ = identify.FinishGetBOS();
    }

    // Setup the initial extensions storage: a chain going to a sort with a stream in the middle for writing.
    util::stream::Chain extensions_chain(util::stream::ChainConfig(sizeof(InitialExtension), 2, config.extension_write_chain_mem));
    util::stream::Stream extensions_write(extensions_chain.Add());
    extensions_first_.reset(new ExtensionsFirstIteration(tuning_words.size(), model_names.size(), max_order, extensions_chain, config.sort));

    // Populate the ContextMap from contexts to instances.
    ContextMap cmap;
    util::FixedArray<InstanceMatch> instances(tuning_words.size());
    {
      UTIL_THROW_IF2(tuning_words.empty(), "Empty tuning data");
      const WordIndex eos = tuning_words.back();
      std::vector<WordIndex> context;
      context.push_back(bos_);
      for (std::size_t i = 0; i < tuning_words.size(); ++i) {
        instances.push_back(boost::ref(extensions_first_->WriteBackoffs(i)), tuning_words[i]);
        for (std::size_t j = 0; j < context.size(); ++j) {
          cmap[util::MurmurHashNative(&context[j], sizeof(WordIndex) * (context.size() - j))].Register(instances.back());
        }
        // Prepare for next word by starting a new sentence or shifting context.
        if (tuning_words[i] == eos) {
          context.clear();
          context.push_back(bos_);
        } else {
          if (context.size() == max_order) {
            context.erase(context.begin());
          }
          context.push_back(tuning_words[i]);
        }
      }
    }

    // Go through each model.  Populate:
    // ln_backoffs_
    ln_backoffs_.resize(instances.size(), models.size());
    // neg_ln_correct_sum_
    neg_ln_correct_sum_.resize(models.size());
    // ln_unigrams_
    ln_unigrams_.resize(combined_vocab_size, models.size());
    // The backoffs in extensions_first_
    for (std::size_t m = 0; m < models.size(); ++m) {
      std::cerr << "Processing model " << m << '/' << models.size() << ": " << model_names[m] << std::endl;
      util::stream::Chains chains(models[m].Order());
      for (std::size_t i = 0; i < models[m].Order(); ++i) {
        // TODO: stop wasting space for backoffs of highest order.
        chains.push_back(util::stream::ChainConfig(NGram<ProbBackoff>::TotalSize(i + 1), 2, config.model_read_chain_mem));
      }
      chains.back().ActivateProgress();
      models[m].Source(chains);
      for (std::size_t i = 0; i < models[m].Order(); ++i) {
        chains[i] >> Renumber(vocab.Mapping(m), i + 1);
      }

      // Populate ln_unigrams_.
      chains[0] >> ReadUnigrams(ln_unigrams_.col(m));

      // Send extensions into extensions_first_ and give data to the instances about backoffs/extensions.
      chains >> JointOrderCallback(m, models[m].Order() - 1, cmap, extensions_write, instances.begin());

      chains >> util::stream::kRecycle;
      chains.Wait(true);
      neg_ln_correct_sum_(m) = 0.0;
      for (InstanceMatch *i = instances.begin(); i != instances.end(); ++i) {
        neg_ln_correct_sum_(m) -= i->FinishModel(m, ln_unigrams_(i->Correct(), m));
        ln_backoffs_(i - instances.begin(), m) = extensions_first_->FullBackoff(i - instances.begin(), m);
      }
      ln_unigrams_(bos_, m) = 0; // Does not matter as long as it does not produce nans since tune_derivatives will overwrite the output.
    }
    extensions_write.Poison();
  }
  extensions_first_->Merge(config.lazy_memory);
}

Instances::~Instances() {}

// TODO: size reduction by excluding order for subsequent passes.
std::size_t Instances::ReadExtensionsEntrySize() const {
  return sizeof(InitialExtension);
}

void Instances::ReadExtensions(util::stream::Chain &on) {
  if (extensions_first_.get()) {
    // Lazy sort and save a sorted copy to disk.  TODO: cut down on record size by stripping out order information.
    extensions_first_->Output(on);
    extensions_first_.reset(); // Relevant data will continue to live in workers.
    extensions_subsequent_.reset(new util::stream::FileBuffer(util::MakeTemp(temp_prefix_)));
    on >> extensions_subsequent_->Sink();
  } else {
    on.SetProgressTarget(extensions_subsequent_->Size());
    on >> extensions_subsequent_->Source();
  }
}

// Back door.
Instances::Instances() {}

}} // namespaces
