// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Generates a random FST according to a class-specific transition model.

#include <cstdlib>
#include <ctime>
#include <memory>
#include <string>

#include <fst/extensions/compress/randmod.h>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/fstlib.h>
#include <fst/util.h>

DEFINE_int32(seed, time(0), "Random seed");
DEFINE_int32(states, 10, "# of states");
DEFINE_int32(labels, 2, "# of labels");
DEFINE_int32(classes, 1, "# of probability distributions");
DEFINE_bool(transducer, false, "Output a transducer");
DEFINE_bool(weights, false, "Output a weighted FST");

int main(int argc, char **argv) {
  using fst::StdVectorFst;
  using fst::StdArc;
  using fst::TropicalWeight;
  using fst::WeightGenerate;

  string usage = "Generates a random FST.\n\n  Usage: ";
  usage += argv[0];
  usage += "[out.fst]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 2) {
    ShowUsage();
    return 1;
  }

  string out_name = (argc > 1 && (strcmp(argv[1], "-") != 0)) ? argv[1] : "";

  srand(FLAGS_seed);

  int num_states = (rand() % FLAGS_states) + 1;    // NOLINT
  int num_classes = (rand() % FLAGS_classes) + 1;  // NOLINT
  int num_labels = (rand() % FLAGS_labels) + 1;    // NOLINT

  StdVectorFst fst;
  using TropicalWeightGenerate = WeightGenerate<TropicalWeight>;
  std::unique_ptr<TropicalWeightGenerate> generate(FLAGS_weights ?
      new TropicalWeightGenerate(false) : nullptr);
  fst::RandMod<StdArc, TropicalWeightGenerate> rand_mod(num_states,
      num_classes, num_labels, FLAGS_transducer, generate.get());
  rand_mod.Generate(&fst);
  fst.Write(out_name);
  return 0;
}
