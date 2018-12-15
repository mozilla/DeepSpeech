// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Regression test for FST weights.

#include <cstdlib>
#include <ctime>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/expectation-weight.h>
#include <fst/float-weight.h>
#include <fst/lexicographic-weight.h>
#include <fst/power-weight.h>
#include <fst/product-weight.h>
#include <fst/set-weight.h>
#include <fst/signed-log-weight.h>
#include <fst/sparse-power-weight.h>
#include <fst/string-weight.h>
#include <fst/union-weight.h>
#include <fst/test/weight-tester.h>

DEFINE_int32(seed, -1, "random seed");
DEFINE_int32(repeat, 10000, "number of test repetitions");

namespace {

using fst::Adder;
using fst::ExpectationWeight;
using fst::GALLIC;
using fst::GallicWeight;
using fst::LexicographicWeight;
using fst::LogWeight;
using fst::LogWeightTpl;
using fst::MinMaxWeight;
using fst::MinMaxWeightTpl;
using fst::NaturalLess;
using fst::PowerWeight;
using fst::ProductWeight;
using fst::SetWeight;
using fst::SET_INTERSECT_UNION;
using fst::SET_UNION_INTERSECT;
using fst::SET_BOOLEAN;
using fst::SignedLogWeight;
using fst::SignedLogWeightTpl;
using fst::SparsePowerWeight;
using fst::StringWeight;
using fst::STRING_LEFT;
using fst::STRING_RIGHT;
using fst::TropicalWeight;
using fst::TropicalWeightTpl;
using fst::UnionWeight;
using fst::WeightConvert;
using fst::WeightGenerate;
using fst::WeightTester;

template <class T>
void TestTemplatedWeights(int repeat) {
  using TropicalWeightGenerate = WeightGenerate<TropicalWeightTpl<T>>;
  TropicalWeightGenerate tropical_generate;
  WeightTester<TropicalWeightTpl<T>, TropicalWeightGenerate> tropical_tester(
      tropical_generate);
  tropical_tester.Test(repeat);

  using LogWeightGenerate = WeightGenerate<LogWeightTpl<T>>;
  LogWeightGenerate log_generate;
  WeightTester<LogWeightTpl<T>, LogWeightGenerate> log_tester(log_generate);
  log_tester.Test(repeat);

  using MinMaxWeightGenerate = WeightGenerate<MinMaxWeightTpl<T>>;
  MinMaxWeightGenerate minmax_generate(true);
  WeightTester<MinMaxWeightTpl<T>, MinMaxWeightGenerate> minmax_tester(
      minmax_generate);
  minmax_tester.Test(repeat);

  using SignedLogWeightGenerate = WeightGenerate<SignedLogWeightTpl<T>>;
  SignedLogWeightGenerate signedlog_generate;
  WeightTester<SignedLogWeightTpl<T>, SignedLogWeightGenerate>
      signedlog_tester(signedlog_generate);
  signedlog_tester.Test(repeat);
}

template <class Weight>
void TestAdder(int n) {
  Weight sum = Weight::Zero();
  Adder<Weight> adder;
  for (int i = 0; i < n; ++i) {
    sum = Plus(sum, Weight::One());
    adder.Add(Weight::One());
  }
  CHECK(ApproxEqual(sum, adder.Sum()));
}

template <class Weight>
void TestSignedAdder(int n) {
  Weight sum = Weight::Zero();
  Adder<Weight> adder;
  const Weight minus_one = Minus(Weight::Zero(), Weight::One());
  for (int i = 0; i < n; ++i) {
    if (i < n/4 || i > 3*n/4) {
      sum = Plus(sum, Weight::One());
      adder.Add(Weight::One());
    } else {
      sum = Minus(sum, Weight::One());
      adder.Add(minus_one);
    }
  }
  CHECK(ApproxEqual(sum, adder.Sum()));
}

template <typename Weight1, typename Weight2>
void TestWeightConversion(Weight1 w1) {
  // Tests round-trp conversion.
  WeightConvert<Weight2, Weight1> to_w1_;
  WeightConvert<Weight1, Weight2> to_w2_;
  Weight2 w2 = to_w2_(w1);
  Weight1 nw1 = to_w1_(w2);
  CHECK_EQ(w1, nw1);
}

template <typename FromWeight, typename ToWeight>
void TestWeightCopy(FromWeight w) {
  // Test copy constructor.
  const ToWeight to_copied(w);
  const FromWeight roundtrip_copied(to_copied);
  CHECK_EQ(w, roundtrip_copied);

  // Test copy assign.
  ToWeight to_copy_assigned;
  to_copy_assigned = w;
  CHECK_EQ(to_copied, to_copy_assigned);

  FromWeight roundtrip_copy_assigned;
  roundtrip_copy_assigned = to_copy_assigned;
  CHECK_EQ(w, roundtrip_copy_assigned);
}

template <typename FromWeight, typename ToWeight>
void TestWeightMove(FromWeight w) {
  // Assume FromWeight -> FromWeight copy works.
  const FromWeight orig(w);
  ToWeight to_moved(std::move(w));
  const FromWeight roundtrip_moved(std::move(to_moved));
  CHECK_EQ(orig, roundtrip_moved);

  // Test move assign.
  w = orig;
  ToWeight to_move_assigned;
  to_move_assigned = std::move(w);
  FromWeight roundtrip_move_assigned;
  roundtrip_move_assigned = std::move(to_move_assigned);
  CHECK_EQ(orig, roundtrip_move_assigned);
}

template <class Weight>
void TestImplicitConversion() {
  // Only test a few of the operations; assumes they are implemented with the
  // same pattern.
  CHECK(Weight(2.0f) == 2.0f);
  CHECK(Weight(2.0) == 2.0);
  CHECK(2.0f == Weight(2.0f));
  CHECK(2.0 == Weight(2.0));

  CHECK_EQ(Weight::Zero(), Times(Weight::Zero(), 3.0f));
  CHECK_EQ(Weight::Zero(), Times(Weight::Zero(), 3.0));
  CHECK_EQ(Weight::Zero(), Times(3.0, Weight::Zero()));

  CHECK_EQ(Weight(3.0), Plus(Weight::Zero(), 3.0f));
  CHECK_EQ(Weight(3.0), Plus(Weight::Zero(), 3.0));
  CHECK_EQ(Weight(3.0), Plus(3.0, Weight::Zero()));
}

void TestPowerWeightGetSetValue() {
  PowerWeight<LogWeight, 3> w;
  // LogWeight has unspecified initial value, so don't check it.
  w.SetValue(0, LogWeight(2));
  w.SetValue(1, LogWeight(3));
  CHECK_EQ(LogWeight(2), w.Value(0));
  CHECK_EQ(LogWeight(3), w.Value(1));
}

void TestSparsePowerWeightGetSetValue() {
  const LogWeight default_value(17);
  SparsePowerWeight<LogWeight> w;
  w.SetDefaultValue(default_value);

  // All gets should be the default.
  CHECK_EQ(default_value, w.Value(0));
  CHECK_EQ(default_value, w.Value(100));

  // First set should fill first_.
  w.SetValue(10, LogWeight(10));
  CHECK_EQ(LogWeight(10), w.Value(10));
  w.SetValue(10, LogWeight(20));
  CHECK_EQ(LogWeight(20), w.Value(10));

  // Add a smaller index.
  w.SetValue(5, LogWeight(5));
  CHECK_EQ(LogWeight(5), w.Value(5));
  CHECK_EQ(LogWeight(20), w.Value(10));

  // Add some larger indices.
  w.SetValue(30, LogWeight(30));
  CHECK_EQ(LogWeight(5), w.Value(5));
  CHECK_EQ(LogWeight(20), w.Value(10));
  CHECK_EQ(LogWeight(30), w.Value(30));

  w.SetValue(29, LogWeight(29));
  CHECK_EQ(LogWeight(5), w.Value(5));
  CHECK_EQ(LogWeight(20), w.Value(10));
  CHECK_EQ(LogWeight(29), w.Value(29));
  CHECK_EQ(LogWeight(30), w.Value(30));

  w.SetValue(31, LogWeight(31));
  CHECK_EQ(LogWeight(5), w.Value(5));
  CHECK_EQ(LogWeight(20), w.Value(10));
  CHECK_EQ(LogWeight(29), w.Value(29));
  CHECK_EQ(LogWeight(30), w.Value(30));
  CHECK_EQ(LogWeight(31), w.Value(31));

  // Replace a value.
  w.SetValue(30, LogWeight(60));
  CHECK_EQ(LogWeight(60), w.Value(30));

  // Replace a value with the default.
  CHECK_EQ(5, w.Size());
  w.SetValue(30, default_value);
  CHECK_EQ(default_value, w.Value(30));
  CHECK_EQ(4, w.Size());

  // Replace lowest index by the default value.
  w.SetValue(5, default_value);
  CHECK_EQ(default_value, w.Value(5));
  CHECK_EQ(3, w.Size());

  // Clear out everything.
  w.SetValue(31, default_value);
  w.SetValue(29, default_value);
  w.SetValue(10, default_value);
  CHECK_EQ(0, w.Size());

  CHECK_EQ(default_value, w.Value(5));
  CHECK_EQ(default_value, w.Value(10));
  CHECK_EQ(default_value, w.Value(29));
  CHECK_EQ(default_value, w.Value(30));
  CHECK_EQ(default_value, w.Value(31));
}

}  // namespace

int main(int argc, char **argv) {
  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(argv[0], &argc, &argv, true);

  LOG(INFO) << "Seed = " << FLAGS_seed;
  srand(FLAGS_seed);

  TestTemplatedWeights<float>(FLAGS_repeat);
  TestTemplatedWeights<double>(FLAGS_repeat);
  FLAGS_fst_weight_parentheses = "()";
  TestTemplatedWeights<float>(FLAGS_repeat);
  TestTemplatedWeights<double>(FLAGS_repeat);
  FLAGS_fst_weight_parentheses = "";

  // Makes sure type names for templated weights are consistent.
  CHECK(TropicalWeight::Type() == "tropical");
  CHECK(TropicalWeightTpl<double>::Type() != TropicalWeightTpl<float>::Type());
  CHECK(LogWeight::Type() == "log");
  CHECK(LogWeightTpl<double>::Type() != LogWeightTpl<float>::Type());
  TropicalWeightTpl<double> w(2.0);
  TropicalWeight tw(2.0);

  TestAdder<TropicalWeight>(1000);
  TestAdder<LogWeight>(1000);
  TestSignedAdder<SignedLogWeight>(1000);

  TestImplicitConversion<LogWeight>();
  TestImplicitConversion<TropicalWeight>();
  TestImplicitConversion<MinMaxWeight>();

  TestWeightConversion<TropicalWeight, LogWeight>(2.0);

  using LeftStringWeight = StringWeight<int>;
  using LeftStringWeightGenerate = WeightGenerate<LeftStringWeight>;
  LeftStringWeightGenerate left_string_generate;
  WeightTester<LeftStringWeight, LeftStringWeightGenerate> left_string_tester(
      left_string_generate);
  left_string_tester.Test(FLAGS_repeat);

  using RightStringWeight = StringWeight<int, STRING_RIGHT>;
  using RightStringWeightGenerate = WeightGenerate<RightStringWeight>;
  RightStringWeightGenerate right_string_generate;
  WeightTester<RightStringWeight, RightStringWeightGenerate>
      right_string_tester(right_string_generate);
  right_string_tester.Test(FLAGS_repeat);

  // STRING_RESTRICT not tested since it requires equal strings,
  // so would fail.

  using IUSetWeight = SetWeight<int, SET_INTERSECT_UNION>;
  using IUSetWeightGenerate = WeightGenerate<IUSetWeight>;
  IUSetWeightGenerate iu_set_generate;
  WeightTester<IUSetWeight, IUSetWeightGenerate>
      iu_set_tester(iu_set_generate);
  iu_set_tester.Test(FLAGS_repeat);

  using UISetWeight = SetWeight<int, SET_UNION_INTERSECT>;
  using UISetWeightGenerate = WeightGenerate<UISetWeight>;
  UISetWeightGenerate ui_set_generate;
  WeightTester<UISetWeight, UISetWeightGenerate>
      ui_set_tester(ui_set_generate);
  ui_set_tester.Test(FLAGS_repeat);

  // SET_INTERSECT_UNION_RESTRICT not tested since it requires equal sets,
  // so would fail.

  using BoolSetWeight = SetWeight<int, SET_BOOLEAN>;
  using BoolSetWeightGenerate = WeightGenerate<BoolSetWeight>;
  BoolSetWeightGenerate bool_set_generate;
  WeightTester<BoolSetWeight, BoolSetWeightGenerate>
      bool_set_tester(bool_set_generate);
  bool_set_tester.Test(FLAGS_repeat);

  TestWeightConversion<IUSetWeight, UISetWeight>(iu_set_generate());

  TestWeightCopy<IUSetWeight, UISetWeight>(iu_set_generate());
  TestWeightCopy<IUSetWeight, BoolSetWeight>(iu_set_generate());
  TestWeightCopy<UISetWeight, IUSetWeight>(ui_set_generate());
  TestWeightCopy<UISetWeight, BoolSetWeight>(ui_set_generate());
  TestWeightCopy<BoolSetWeight, IUSetWeight>(bool_set_generate());
  TestWeightCopy<BoolSetWeight, UISetWeight>(bool_set_generate());

  TestWeightMove<IUSetWeight, UISetWeight>(iu_set_generate());
  TestWeightMove<IUSetWeight, BoolSetWeight>(iu_set_generate());
  TestWeightMove<UISetWeight, IUSetWeight>(ui_set_generate());
  TestWeightMove<UISetWeight, BoolSetWeight>(ui_set_generate());
  TestWeightMove<BoolSetWeight, IUSetWeight>(bool_set_generate());
  TestWeightMove<BoolSetWeight, UISetWeight>(bool_set_generate());

  // COMPOSITE WEIGHTS AND TESTERS - DEFINITIONS

  using TropicalGallicWeight = GallicWeight<int, TropicalWeight>;
  using TropicalGallicWeightGenerate = WeightGenerate<TropicalGallicWeight>;
  TropicalGallicWeightGenerate tropical_gallic_generate(true);
  WeightTester<TropicalGallicWeight, TropicalGallicWeightGenerate>
      tropical_gallic_tester(tropical_gallic_generate);

  using TropicalGenGallicWeight = GallicWeight<int, TropicalWeight, GALLIC>;
  using TropicalGenGallicWeightGenerate =
      WeightGenerate<TropicalGenGallicWeight>;
  TropicalGenGallicWeightGenerate tropical_gen_gallic_generate(false);
  WeightTester<TropicalGenGallicWeight, TropicalGenGallicWeightGenerate>
      tropical_gen_gallic_tester(tropical_gen_gallic_generate);

  using TropicalProductWeight = ProductWeight<TropicalWeight, TropicalWeight>;
  using TropicalProductWeightGenerate = WeightGenerate<TropicalProductWeight>;
  TropicalProductWeightGenerate tropical_product_generate;
  WeightTester<TropicalProductWeight, TropicalProductWeightGenerate>
      tropical_product_tester(tropical_product_generate);

  using TropicalLexicographicWeight =
      LexicographicWeight<TropicalWeight, TropicalWeight>;
  using TropicalLexicographicWeightGenerate =
      WeightGenerate<TropicalLexicographicWeight>;
  TropicalLexicographicWeightGenerate tropical_lexicographic_generate;
  WeightTester<TropicalLexicographicWeight,
               TropicalLexicographicWeightGenerate>
      tropical_lexicographic_tester(tropical_lexicographic_generate);

  using TropicalCubeWeight = PowerWeight<TropicalWeight, 3>;
  using TropicalCubeWeightGenerate = WeightGenerate<TropicalCubeWeight>;
  TropicalCubeWeightGenerate tropical_cube_generate;
  WeightTester<TropicalCubeWeight, TropicalCubeWeightGenerate>
      tropical_cube_tester(tropical_cube_generate);

  using FirstNestedProductWeight =
      ProductWeight<TropicalProductWeight, TropicalWeight>;
  using FirstNestedProductWeightGenerate =
      WeightGenerate<FirstNestedProductWeight>;
  FirstNestedProductWeightGenerate first_nested_product_generate;
  WeightTester<FirstNestedProductWeight, FirstNestedProductWeightGenerate>
      first_nested_product_tester(first_nested_product_generate);

  using SecondNestedProductWeight =
      ProductWeight<TropicalWeight, TropicalProductWeight>;
  using SecondNestedProductWeightGenerate =
      WeightGenerate<SecondNestedProductWeight>;
  SecondNestedProductWeightGenerate second_nested_product_generate;
  WeightTester<SecondNestedProductWeight, SecondNestedProductWeightGenerate>
      second_nested_product_tester(second_nested_product_generate);

  using NestedProductCubeWeight = PowerWeight<FirstNestedProductWeight, 3>;
  using NestedProductCubeWeightGenerate =
      WeightGenerate<NestedProductCubeWeight>;
  NestedProductCubeWeightGenerate nested_product_cube_generate;
  WeightTester<NestedProductCubeWeight, NestedProductCubeWeightGenerate>
      nested_product_cube_tester(nested_product_cube_generate);

  using SparseNestedProductCubeWeight =
      SparsePowerWeight<NestedProductCubeWeight, size_t>;
  using SparseNestedProductCubeWeightGenerate =
      WeightGenerate<SparseNestedProductCubeWeight>;
  SparseNestedProductCubeWeightGenerate sparse_nested_product_cube_generate;
  WeightTester<SparseNestedProductCubeWeight,
               SparseNestedProductCubeWeightGenerate>
      sparse_nested_product_cube_tester(sparse_nested_product_cube_generate);

  using LogSparsePowerWeight = SparsePowerWeight<LogWeight, size_t>;
  using LogSparsePowerWeightGenerate = WeightGenerate<LogSparsePowerWeight>;
  LogSparsePowerWeightGenerate log_sparse_power_generate;
  WeightTester<LogSparsePowerWeight, LogSparsePowerWeightGenerate>
      log_sparse_power_tester(log_sparse_power_generate);

  using LogLogExpectationWeight = ExpectationWeight<LogWeight, LogWeight>;
  using LogLogExpectationWeightGenerate =
      WeightGenerate<LogLogExpectationWeight>;
  LogLogExpectationWeightGenerate log_log_expectation_generate;
  WeightTester<LogLogExpectationWeight, LogLogExpectationWeightGenerate>
      log_log_expectation_tester(log_log_expectation_generate);

  using LogLogSparseExpectationWeight =
      ExpectationWeight<LogWeight, LogSparsePowerWeight>;
  using LogLogSparseExpectationWeightGenerate =
      WeightGenerate<LogLogSparseExpectationWeight>;
  LogLogSparseExpectationWeightGenerate log_log_sparse_expectation_generate;
  WeightTester<LogLogSparseExpectationWeight,
               LogLogSparseExpectationWeightGenerate>
      log_log_sparse_expectation_tester(log_log_sparse_expectation_generate);

  struct UnionWeightOptions {
    using Compare = NaturalLess<TropicalWeight>;

    struct Merge {
      TropicalWeight operator()(const TropicalWeight &w1,
                                const TropicalWeight &w2) const {
        return w1;
      }
    };

    using ReverseOptions = UnionWeightOptions;
  };

  using TropicalUnionWeight = UnionWeight<TropicalWeight, UnionWeightOptions>;
  using TropicalUnionWeightGenerate = WeightGenerate<TropicalUnionWeight>;
  TropicalUnionWeightGenerate tropical_union_generate;
  WeightTester<TropicalUnionWeight, TropicalUnionWeightGenerate>
      tropical_union_tester(tropical_union_generate);

  // COMPOSITE WEIGHTS AND TESTERS - TESTING

  // Tests composite weight I/O with parentheses.
  FLAGS_fst_weight_parentheses = "()";

  // Unnested composite.
  tropical_gallic_tester.Test(FLAGS_repeat);
  tropical_gen_gallic_tester.Test(FLAGS_repeat);
  tropical_product_tester.Test(FLAGS_repeat);
  tropical_lexicographic_tester.Test(FLAGS_repeat);
  tropical_cube_tester.Test(FLAGS_repeat);
  log_sparse_power_tester.Test(FLAGS_repeat);
  log_log_expectation_tester.Test(FLAGS_repeat, false);
  tropical_union_tester.Test(FLAGS_repeat, false);

  // Nested composite.
  first_nested_product_tester.Test(FLAGS_repeat);
  second_nested_product_tester.Test(5);
  nested_product_cube_tester.Test(FLAGS_repeat);
  sparse_nested_product_cube_tester.Test(FLAGS_repeat);
  log_log_sparse_expectation_tester.Test(FLAGS_repeat, false);

  // ... and tests composite weight I/O without parentheses.
  FLAGS_fst_weight_parentheses = "";

  // Unnested composite.
  tropical_gallic_tester.Test(FLAGS_repeat);
  tropical_product_tester.Test(FLAGS_repeat);
  tropical_lexicographic_tester.Test(FLAGS_repeat);
  tropical_cube_tester.Test(FLAGS_repeat);
  log_sparse_power_tester.Test(FLAGS_repeat);
  log_log_expectation_tester.Test(FLAGS_repeat, false);
  tropical_union_tester.Test(FLAGS_repeat, false);

  // Nested composite.
  second_nested_product_tester.Test(FLAGS_repeat);
  log_log_sparse_expectation_tester.Test(FLAGS_repeat, false);

  TestPowerWeightGetSetValue();
  TestSparsePowerWeightGetSetValue();

  std::cout << "PASS" << std::endl;

  return 0;
}
