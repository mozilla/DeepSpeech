#include "lm/common/compare.hh"
#include "lm/common/model_buffer.hh"
#include "lm/common/ngram.hh"
#include "util/stream/chain.hh"
#include "util/stream/multi_stream.hh"
#include "util/stream/sort.hh"
#include "lm/interpolate/split_worker.hh"

#include <boost/program_options.hpp>
#include <boost/version.hpp>

#if defined(_WIN32) || defined(_WIN64)

// Windows doesn't define <unistd.h>
//
// So we define what we need here instead:
//
#define STDIN_FILENO = 0
#define STDOUT_FILENO = 1
#else // Huzzah for POSIX!
#include <unistd.h>
#endif

/*
 * This is a simple example program that takes in intermediate
 * suffix-sorted ngram files and outputs two sets of files: one for backoff
 * probability values (raw numbers, in suffix order) and one for
 * probability values (ngram id and probability, in *context* order)
 */
int main(int argc, char *argv[]) {
  using namespace lm::interpolate;

  const std::size_t ONE_GB = 1 << 30;
  const std::size_t SIXTY_FOUR_MB = 1 << 26;
  const std::size_t NUMBER_OF_BLOCKS = 2;

  std::string FILE_NAME = "ngrams";
  std::string CONTEXT_SORTED_FILENAME = "csorted-ngrams";
  std::string BACKOFF_FILENAME = "backoffs";
  std::string TMP_DIR = "/tmp/";

  try {
    namespace po = boost::program_options;
    po::options_description options("canhazinterp Pass-3 options");

    options.add_options()
      ("help,h", po::bool_switch(), "Show this help message")
      ("ngrams,n", po::value<std::string>(&FILE_NAME), "ngrams file")
      ("csortngrams,c", po::value<std::string>(&CONTEXT_SORTED_FILENAME), "context sorted ngrams file")
      ("backoffs,b", po::value<std::string>(&BACKOFF_FILENAME), "backoffs file")
      ("tmpdir,t", po::value<std::string>(&TMP_DIR), "tmp dir");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);

    // Display help
    if(vm["help"].as<bool>()) {
      std::cerr << "Usage: " << options << std::endl;
      return 1;
    }
  }
  catch(const std::exception &e) {

    std::cerr << e.what() << std::endl;
    return 1;

  }

  // The basic strategy here is to have three chains:
  // - The first reads the ngram order inputs using ModelBuffer. Those are
  //   then stripped of their backoff values and fed into the third chain;
  //   the backoff values *themselves* are written to the second chain.
  //
  // - The second chain takes the backoff values and writes them out to a
  //   file (one for each order).
  //
  // - The third chain takes just the probability values and ngrams and
  //   writes them out, sorted in context-order, to a file (one for each
  //   order).

  // This will be used to read in the binary intermediate files. There is
  // one file per order (e.g. ngrams.1, ngrams.2, ...)
  lm::ModelBuffer buffer(FILE_NAME);

  // Create a separate chains for each ngram order for:
  // - Input from the intermediate files
  // - Output to the backoff file
  // - Output to the (context-sorted) probability file
  util::stream::Chains ngram_inputs(buffer.Order());
  util::stream::Chains backoff_chains(buffer.Order());
  util::stream::Chains prob_chains(buffer.Order());
  for (std::size_t i = 0; i < buffer.Order(); ++i) {
    ngram_inputs.push_back(util::stream::ChainConfig(
        lm::NGram<lm::ProbBackoff>::TotalSize(i + 1), NUMBER_OF_BLOCKS, ONE_GB));

    backoff_chains.push_back(
        util::stream::ChainConfig(sizeof(float), NUMBER_OF_BLOCKS, ONE_GB));

    prob_chains.push_back(util::stream::ChainConfig(
        sizeof(lm::WordIndex) * (i + 1) + sizeof(float), NUMBER_OF_BLOCKS,
        ONE_GB));
  }

  // This sets the input for each of the ngram order chains to the
  // appropriate file
  buffer.Source(ngram_inputs);

  util::FixedArray<util::scoped_ptr<SplitWorker> > workers(buffer.Order());
  for (std::size_t i = 0; i < buffer.Order(); ++i) {
    // Attach a SplitWorker to each of the ngram input chains, writing to the
    // corresponding order's backoff and probability chains
    workers.push_back(
        new SplitWorker(i + 1, backoff_chains[i], prob_chains[i]));
    ngram_inputs[i] >> boost::ref(*workers.back());
  }

  util::stream::SortConfig sort_cfg;
  sort_cfg.temp_prefix = TMP_DIR;
  sort_cfg.buffer_size = SIXTY_FOUR_MB;
  sort_cfg.total_memory = ONE_GB;

  // This will parallel merge sort the individual order files, putting
  // them in context-order instead of suffix-order.
  //
  // Two new threads will be running, each owned by the chains[i] object.
  // - The first executes BlockSorter.Run() to sort the n-gram entries
  // - The second executes WriteAndRecycle.Run() to write each sorted
  //   block to disk as a temporary file
  util::stream::Sorts<lm::ContextOrder> sorts(buffer.Order());
  for (std::size_t i = 0; i < prob_chains.size(); ++i) {
    sorts.push_back(prob_chains[i], sort_cfg, lm::ContextOrder(i + 1));
  }

  // Set the sort output to be on the same chain
  for (std::size_t i = 0; i < prob_chains.size(); ++i) {
    // The following call to Chain::Wait()
    //     joins the threads owned by chains[i].
    //
    // As such the following call won't return
    //     until all threads owned by chains[i] have completed.
    //
    // The following call also resets chain[i]
    //     so that it can be reused
    //     (including free'ing the memory previously used by the chain)
    prob_chains[i].Wait();

    // In an ideal world (without memory restrictions)
    //     we could merge all of the previously sorted blocks
    //     by reading them all completely into memory
    //     and then running merge sort over them.
    //
    // In the real world, we have memory restrictions;
    //     depending on how many blocks we have,
    //     and how much memory we can use to read from each block
    //     (sort_config.buffer_size)
    //     it may be the case that we have insufficient memory
    //     to read sort_config.buffer_size of data from each block from disk.
    //
    // If this occurs, then it will be necessary to perform one or more rounds
    // of merge sort on disk;
    //     doing so will reduce the number of blocks that we will eventually
    //     need to read from
    //     when performing the final round of merge sort in memory.
    //
    // So, the following call determines whether it is necessary
    //     to perform one or more rounds of merge sort on disk;
    //     if such on-disk merge sorting is required, such sorting is performed.
    //
    // Finally, the following method launches a thread that calls
    // OwningMergingReader.Run()
    //     to perform the final round of merge sort in memory.
    //
    // Merge sort could have be invoked directly
    //     so that merge sort memory doesn't coexist with Chain memory.
    sorts[i].Output(prob_chains[i]);
  }

  // Create another model buffer for our output on e.g. csorted-ngrams.1,
  // csorted-ngrams.2, ...
  lm::ModelBuffer output_buf(CONTEXT_SORTED_FILENAME, true, false);
  output_buf.Sink(prob_chains, buffer.Counts());

  // Create a third model buffer for our backoff output on e.g. backoff.1,
  // backoff.2, ...
  lm::ModelBuffer boff_buf(BACKOFF_FILENAME, true, false);
  boff_buf.Sink(backoff_chains, buffer.Counts());

  // Joins all threads that chains owns,
  //    and does a for loop over each chain object in chains,
  //    calling chain.Wait() on each such chain object
  ngram_inputs.Wait(true);
  backoff_chains.Wait(true);
  prob_chains.Wait(true);

  return 0;
}
