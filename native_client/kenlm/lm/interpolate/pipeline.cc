#include "lm/interpolate/pipeline.hh"

#include "lm/common/compare.hh"
#include "lm/common/print.hh"
#include "lm/common/renumber.hh"
#include "lm/vocab.hh"
#include "lm/interpolate/backoff_reunification.hh"
#include "lm/interpolate/interpolate_info.hh"
#include "lm/interpolate/merge_probabilities.hh"
#include "lm/interpolate/merge_vocab.hh"
#include "lm/interpolate/normalize.hh"
#include "lm/interpolate/universal_vocab.hh"
#include "util/stream/chain.hh"
#include "util/stream/count_records.hh"
#include "util/stream/io.hh"
#include "util/stream/multi_stream.hh"
#include "util/stream/sort.hh"
#include "util/fixed_array.hh"

namespace lm { namespace interpolate { namespace {

/* Put the original input files on chains and renumber them */
void SetupInputs(std::size_t buffer_size, const UniversalVocab &vocab, util::FixedArray<ModelBuffer> &models, bool exclude_highest, util::FixedArray<util::stream::Chains> &chains, util::FixedArray<util::stream::ChainPositions> &positions) {
  chains.clear();
  positions.clear();
  // TODO: much better memory sizing heuristics e.g. not making the chain larger than it will use.
  util::stream::ChainConfig config(0, 2, buffer_size);
  for (std::size_t i = 0; i < models.size(); ++i) {
    chains.push_back(models[i].Order() - exclude_highest);
    for (std::size_t j = 0; j < models[i].Order() - exclude_highest; ++j) {
      config.entry_size = sizeof(WordIndex) * (j + 1) + sizeof(float) * 2; // TODO do not include wasteful backoff for highest.
      chains.back().push_back(config);
    }
    if (i == models.size() - 1)
      chains.back().back().ActivateProgress();
    models[i].Source(chains.back());
    for (std::size_t j = 0; j < models[i].Order() - exclude_highest; ++j) {
      chains[i][j] >> Renumber(vocab.Mapping(i), j + 1);
    }
  }
 for (std::size_t i = 0; i < chains.size(); ++i) {
    positions.push_back(chains[i]);
  }
}

template <class Compare> void SinkSort(const util::stream::SortConfig &config, util::stream::Chains &chains, util::stream::Sorts<Compare> &sorts) {
  for (std::size_t i = 0; i < chains.size(); ++i) {
    sorts.push_back(chains[i], config, Compare(i + 1));
  }
}

template <class Compare> void SourceSort(util::stream::Chains &chains, util::stream::Sorts<Compare> &sorts) {
  // TODO memory management
  for (std::size_t i = 0; i < sorts.size(); ++i) {
    sorts[i].Merge(sorts[i].DefaultLazy());
  }
  for (std::size_t i = 0; i < sorts.size(); ++i) {
    sorts[i].Output(chains[i], sorts[i].DefaultLazy());
  }
}

} // namespace

void Pipeline(util::FixedArray<ModelBuffer> &models, const Config &config, int write_file) {
  // Setup InterpolateInfo and UniversalVocab.
  InterpolateInfo info;
  info.lambdas = config.lambdas;
  std::vector<WordIndex> vocab_sizes;

  util::scoped_fd vocab_null(util::MakeTemp(config.sort.temp_prefix));
  std::size_t max_order = 0;
  util::FixedArray<int> vocab_files(models.size());
  for (ModelBuffer *i = models.begin(); i != models.end(); ++i) {
    info.orders.push_back(i->Order());
    vocab_sizes.push_back(i->Counts()[0]);
    vocab_files.push_back(i->VocabFile());
    max_order = std::max(max_order, i->Order());
  }
  util::scoped_ptr<UniversalVocab> vocab(new UniversalVocab(vocab_sizes));
  {
    ngram::ImmediateWriteWordsWrapper writer(NULL, vocab_null.get(), 0);
    MergeVocab(vocab_files, *vocab, writer);
  }

  std::cerr << "Merging probabilities." << std::endl;
  // Pass 1: merge probabilities
  util::FixedArray<util::stream::Chains> input_chains(models.size());
  util::FixedArray<util::stream::ChainPositions> models_by_order(models.size());
  SetupInputs(config.BufferSize(), *vocab, models, false, input_chains, models_by_order);

  util::stream::Chains merged_probs(max_order);
  for (std::size_t i = 0; i < max_order; ++i) {
    merged_probs.push_back(util::stream::ChainConfig(PartialProbGamma::TotalSize(info, i + 1), 2, config.BufferSize())); // TODO: not buffer_size
  }
  merged_probs >> MergeProbabilities(info, models_by_order);
  std::vector<uint64_t> counts(max_order);
  for (std::size_t i = 0; i < max_order; ++i) {
    merged_probs[i] >> util::stream::CountRecords(&counts[i]);
  }
  for (util::stream::Chains *i = input_chains.begin(); i != input_chains.end(); ++i) {
    *i >> util::stream::kRecycle;
  }

  // Pass 2: normalize.
  {
    util::stream::Sorts<ContextOrder> sorts(merged_probs.size());
    SinkSort(config.sort, merged_probs, sorts);
    merged_probs.Wait(true);
    for (util::stream::Chains *i = input_chains.begin(); i != input_chains.end(); ++i) {
      i->Wait(true);
    }
    SourceSort(merged_probs, sorts);
  }

  std::cerr << "Normalizing" << std::endl;
  SetupInputs(config.BufferSize(), *vocab, models, true, input_chains, models_by_order);
  util::stream::Chains probabilities(max_order), backoffs(max_order - 1);
  std::size_t block_count = 2;
  for (std::size_t i = 0; i < max_order; ++i) {
    // Careful accounting to ensure RewindableStream can fit the entire vocabulary.
    block_count = std::max<std::size_t>(block_count, 2);
    // This much needs to fit in RewindableStream.
    std::size_t fit = NGram<float>::TotalSize(i + 1) * counts[0];
    // fit / (block_count - 1) rounded up
    std::size_t min_block = (fit + block_count - 2) / (block_count - 1);
    std::size_t specify = std::max(config.BufferSize(), min_block * block_count);
    probabilities.push_back(util::stream::ChainConfig(NGram<float>::TotalSize(i + 1), block_count, specify));
  }
  for (std::size_t i = 0; i < max_order - 1; ++i) {
    backoffs.push_back(util::stream::ChainConfig(sizeof(float), 2, config.BufferSize()));
  }
  Normalize(info, models_by_order, merged_probs, probabilities, backoffs);
  util::FixedArray<util::stream::FileBuffer> backoff_buffers(backoffs.size());
  for (std::size_t i = 0; i < max_order - 1; ++i) {
    backoff_buffers.push_back(util::MakeTemp(config.sort.temp_prefix));
    backoffs[i] >> backoff_buffers.back().Sink() >> util::stream::kRecycle;
  }
  for (util::stream::Chains *i = input_chains.begin(); i != input_chains.end(); ++i) {
    *i >> util::stream::kRecycle;
  }
  merged_probs >> util::stream::kRecycle;

  // Pass 3: backoffs in the right place.
  {
    util::stream::Sorts<SuffixOrder> sorts(probabilities.size());
    SinkSort(config.sort, probabilities, sorts);
    probabilities.Wait(true);
    for (util::stream::Chains *i = input_chains.begin(); i != input_chains.end(); ++i) {
      i->Wait(true);
    }
    backoffs.Wait(true);
    merged_probs.Wait(true);
    // destroy universal vocab to save RAM.
    vocab.reset();
    SourceSort(probabilities, sorts);
  }

  std::cerr << "Reunifying backoffs" << std::endl;
  util::stream::ChainPositions prob_pos(max_order - 1);
  util::stream::Chains combined(max_order - 1);
  for (std::size_t i = 0; i < max_order - 1; ++i) {
    if (i == max_order - 2)
      backoffs[i].ActivateProgress();
    backoffs[i].SetProgressTarget(backoff_buffers[i].Size());
    backoffs[i] >> backoff_buffers[i].Source(true);
    prob_pos.push_back(probabilities[i].Add());
    combined.push_back(util::stream::ChainConfig(NGram<ProbBackoff>::TotalSize(i + 1), 2, config.BufferSize()));
  }
  util::stream::ChainPositions backoff_pos(backoffs);

  ReunifyBackoff(prob_pos, backoff_pos, combined);

  util::stream::ChainPositions output_pos(max_order);
  for (std::size_t i = 0; i < max_order - 1; ++i) {
    output_pos.push_back(combined[i].Add());
  }
  output_pos.push_back(probabilities.back().Add());

  probabilities >> util::stream::kRecycle;
  backoffs >> util::stream::kRecycle;
  combined >> util::stream::kRecycle;

  // TODO genericize to ModelBuffer etc.
  PrintARPA(vocab_null.get(), write_file, counts).Run(output_pos);
}

}} // namespaces
