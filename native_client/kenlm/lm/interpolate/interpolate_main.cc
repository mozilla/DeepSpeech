#include "lm/common/model_buffer.hh"
#include "lm/common/size_option.hh"
#include "lm/interpolate/pipeline.hh"
#include "lm/interpolate/tune_instances.hh"
#include "lm/interpolate/tune_weights.hh"
#include "util/fixed_array.hh"
#include "util/usage.hh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas" // Older gcc doesn't have "-Wunused-local-typedefs" and complains.
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <Eigen/Core>
#pragma GCC diagnostic pop

#include <boost/program_options.hpp>

#include <iostream>
#include <vector>

namespace {
void MungeWeightArgs(int argc, char *argv[], std::vector<const char *> &munged_args) {
  // Boost program options doesn't -w 0.2 -0.1 because it thinks -0.1 is an
  // option.  There appears to be no standard way to fix this without breaking
  // single-dash arguments.  So here's a hack: put a -w before every number
  // if it's within the scope of a weight argument.
  munged_args.push_back(argv[0]);
  char **inside_weights = NULL;
  for (char **i = argv + 1; i < argv + argc; ++i) {
    StringPiece arg(*i);
    if (starts_with(arg, "-w") || starts_with(arg, "--w")) {
      inside_weights = i;
    } else if (inside_weights && arg.size() >= 2 && arg[0] == '-' && ((arg[1] >= '0' && arg[1] <= '9') || arg[1] == '.')) {
      // If a negative number appears right after -w, don't add another -w.
      // And do stay inside weights.
      if (inside_weights + 1 != i) {
        munged_args.push_back("-w");
      }
    } else if (starts_with(arg, "-")) {
      inside_weights = NULL;
    }
    munged_args.push_back(*i);
  }
}
} // namespace

int main(int argc, char *argv[]) {
  try {
    Eigen::initParallel();
    lm::interpolate::Config pipe_config;
    lm::interpolate::InstancesConfig instances_config;
    std::vector<std::string> input_models;
    std::string tuning_file;

    namespace po = boost::program_options;
    po::options_description options("Log-linear interpolation options");
    options.add_options()
      ("help,h", po::bool_switch(), "Show this help message")
      ("model,m", po::value<std::vector<std::string> >(&input_models)->multitoken()->required(), "Models to interpolate, which must be in KenLM intermediate format.  The intermediate format can be generated using the --intermediate argument to lmplz.")
      ("weight,w", po::value<std::vector<float> >(&pipe_config.lambdas)->multitoken(), "Interpolation weights")
      ("tuning,t", po::value<std::string>(&tuning_file), "File to tune on: a text file with one sentence per line")
      ("just_tune", po::bool_switch(), "Tune and print weights then quit")
      ("temp_prefix,T", po::value<std::string>(&pipe_config.sort.temp_prefix)->default_value("/tmp/lm"), "Temporary file prefix")
      ("memory,S", lm::SizeOption(pipe_config.sort.total_memory, util::GuessPhysicalMemory() ? "50%" : "1G"), "Sorting memory: this is a very rough guide")
      ("sort_block", lm::SizeOption(pipe_config.sort.buffer_size, "64M"), "Block size");
    po::variables_map vm;

    std::vector<const char *> munged_args;
    MungeWeightArgs(argc, argv, munged_args);

    po::store(po::parse_command_line((int)munged_args.size(), &*munged_args.begin(), options), vm);
    if (argc == 1 || vm["help"].as<bool>()) {
      std::cerr << "Interpolate multiple models\n" << options << std::endl;
      return 1;
    }
    po::notify(vm);
    instances_config.sort = pipe_config.sort;
    instances_config.model_read_chain_mem = instances_config.sort.buffer_size;
    instances_config.extension_write_chain_mem = instances_config.sort.total_memory;
    instances_config.lazy_memory = instances_config.sort.total_memory;

    if (pipe_config.lambdas.empty() && tuning_file.empty()) {
      std::cerr << "Provide a tuning file with -t xor weights with -w." << std::endl;
      return 1;
    }
    if (!pipe_config.lambdas.empty() && !tuning_file.empty()) {
      std::cerr << "Provide weights xor a tuning file, not both." << std::endl;
      return 1;
    }

    if (!tuning_file.empty()) {
      // Tune weights
      std::vector<StringPiece> model_names;
      for (std::vector<std::string>::const_iterator i = input_models.begin(); i != input_models.end(); ++i) {
        model_names.push_back(*i);
      }
      lm::interpolate::TuneWeights(util::OpenReadOrThrow(tuning_file.c_str()), model_names, instances_config, pipe_config.lambdas);

      std::cerr << "Final weights:";
      std::ostream &to = vm["just_tune"].as<bool>() ? std::cout : std::cerr;
      for (std::vector<float>::const_iterator i = pipe_config.lambdas.begin(); i != pipe_config.lambdas.end(); ++i) {
        to << ' ' << *i;
      }
      to << std::endl;
    }
    if (vm["just_tune"].as<bool>()) {
      return 0;
    }

    if (pipe_config.lambdas.size() != input_models.size()) {
      std::cerr << "Number of models (" << input_models.size() << ") should match the number of weights (" << pipe_config.lambdas.size() << ")." << std::endl;
      return 1;
    }

    util::FixedArray<lm::ModelBuffer> models(input_models.size());
    for (std::size_t i = 0; i < input_models.size(); ++i) {
      models.push_back(input_models[i]);
    }
    lm::interpolate::Pipeline(models, pipe_config, 1);
  } catch (const std::exception &e) {
    std::cerr << e.what() <<std::endl;
    return 1;
  }
  return 0;
}
