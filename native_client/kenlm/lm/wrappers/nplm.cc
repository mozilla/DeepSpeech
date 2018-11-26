#include "lm/wrappers/nplm.hh"
#include "util/exception.hh"
#include "util/file.hh"

#include <algorithm>
#include <cstring>

#include "neuralLM.h"

namespace lm {
namespace np {

Vocabulary::Vocabulary(const nplm::vocabulary &vocab)
  : base::Vocabulary(vocab.lookup_word("<s>"), vocab.lookup_word("</s>"), vocab.lookup_word("<unk>")),
    vocab_(vocab), null_word_(vocab.lookup_word("<null>")) {}

Vocabulary::~Vocabulary() {}

WordIndex Vocabulary::Index(const std::string &str) const {
  return vocab_.lookup_word(str);
}

class Backend {
  public:
    Backend(const nplm::neuralLM &from, const std::size_t cache_size) : lm_(from), ngram_(from.get_order()) {
      lm_.set_cache(cache_size);
    }

    nplm::neuralLM &LM() { return lm_; }
    const nplm::neuralLM &LM() const { return lm_; }

    Eigen::Matrix<int,Eigen::Dynamic,1> &staging_ngram() { return ngram_; }

    double lookup_from_staging() { return lm_.lookup_ngram(ngram_); }

    int order() const { return lm_.get_order(); }

  private:
    nplm::neuralLM lm_;
    Eigen::Matrix<int,Eigen::Dynamic,1> ngram_;
};

bool Model::Recognize(const std::string &name) {
  try {
    util::scoped_fd file(util::OpenReadOrThrow(name.c_str()));
    char magic_check[16];
    util::ReadOrThrow(file.get(), magic_check, sizeof(magic_check));
    const char nnlm_magic[] = "\\config\nversion ";
    return !memcmp(magic_check, nnlm_magic, 16);
  } catch (const util::Exception &) {
    return false;
  }
}

namespace {
nplm::neuralLM *LoadNPLM(const std::string &file) {
  util::scoped_ptr<nplm::neuralLM> ret(new nplm::neuralLM());
  ret->read(file);
  return ret.release();
}
} // namespace

Model::Model(const std::string &file, std::size_t cache)
  : base_instance_(LoadNPLM(file)), vocab_(base_instance_->get_vocabulary()), cache_size_(cache) {
  UTIL_THROW_IF(base_instance_->get_order() > NPLM_MAX_ORDER, util::Exception, "This NPLM has order " << (unsigned int)base_instance_->get_order() << " but the KenLM wrapper was compiled with " << NPLM_MAX_ORDER << ".  Change the defintion of NPLM_MAX_ORDER and recompile.");
  // log10 compatible with backoff models.
  base_instance_->set_log_base(10.0);
  State begin_sentence, null_context;
  std::fill(begin_sentence.words, begin_sentence.words + NPLM_MAX_ORDER - 1, base_instance_->lookup_word("<s>"));
  null_word_ = base_instance_->lookup_word("<null>");
  std::fill(null_context.words, null_context.words + NPLM_MAX_ORDER - 1, null_word_);

  Init(begin_sentence, null_context, vocab_, base_instance_->get_order());
}

Model::~Model() {}

FullScoreReturn Model::FullScore(const State &from, const WordIndex new_word, State &out_state) const {
  Backend *backend = backend_.get();
  if (!backend) {
    backend = new Backend(*base_instance_, cache_size_);
    backend_.reset(backend);
  }
  // State is in natural word order.
  FullScoreReturn ret;
  for (int i = 0; i < backend->order() - 1; ++i) {
    backend->staging_ngram()(i) = from.words[i];
  }
  backend->staging_ngram()(backend->order() - 1) = new_word;
  ret.prob = backend->lookup_from_staging();
  // Always say full order.
  ret.ngram_length = backend->order();
  // Shift everything down by one.
  memcpy(out_state.words, from.words + 1, sizeof(WordIndex) * (backend->order() - 2));
  out_state.words[backend->order() - 2] = new_word;
  // Fill in trailing words with zeros so state comparison works.
  memset(out_state.words + backend->order() - 1, 0, sizeof(WordIndex) * (NPLM_MAX_ORDER - backend->order()));
  return ret;
}

// TODO: optimize with direct call?
FullScoreReturn Model::FullScoreForgotState(const WordIndex *context_rbegin, const WordIndex *context_rend, const WordIndex new_word, State &out_state) const {
  // State is in natural word order.  The API here specifies reverse order.
  std::size_t state_length = std::min<std::size_t>(Order() - 1, context_rend - context_rbegin);
  State state;
  // Pad with null words.
  for (lm::WordIndex *i = state.words; i < state.words + Order() - 1 - state_length; ++i) {
    *i = null_word_;
  }
  // Put new words at the end.
  std::reverse_copy(context_rbegin, context_rbegin + state_length, state.words + Order() - 1 - state_length);
  return FullScore(state, new_word, out_state);
}

} // namespace np
} // namespace lm
