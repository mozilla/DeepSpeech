// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Utility class for regression testing of FST weights.

#ifndef FST_TEST_WEIGHT_TESTER_H_
#define FST_TEST_WEIGHT_TESTER_H_

#include <iostream>
#include <sstream>

#include <utility>

#include <fst/log.h>
#include <fst/weight.h>

namespace fst {

// This class tests a variety of identities and properties that must
// hold for the Weight class to be well-defined. It calls function object
// WEIGHT_GENERATOR to select weights that are used in the tests.
template <class Weight, class WeightGenerator>
class WeightTester {
 public:
  WeightTester(WeightGenerator generator)
      : weight_generator_(std::move(generator)) {}

  void Test(int iterations, bool test_division = true) {
    for (int i = 0; i < iterations; ++i) {
      // Selects the test weights.
      const Weight w1(weight_generator_());
      const Weight w2(weight_generator_());
      const Weight w3(weight_generator_());

      VLOG(1) << "weight type = " << Weight::Type();
      VLOG(1) << "w1 = " << w1;
      VLOG(1) << "w2 = " << w2;
      VLOG(1) << "w3 = " << w3;

      TestSemiring(w1, w2, w3);
      if (test_division) TestDivision(w1, w2);
      TestReverse(w1, w2);
      TestEquality(w1, w2, w3);
      TestIO(w1);
      TestCopy(w1);
    }
  }

 private:
  // Note in the tests below we use ApproxEqual rather than == and add
  // kDelta to inequalities where the weights might be inexact.

  // Tests (Plus, Times, Zero, One) defines a commutative semiring.
  void TestSemiring(Weight w1, Weight w2, Weight w3) {
    // Checks that the operations are closed.
    CHECK(Plus(w1, w2).Member());
    CHECK(Times(w1, w2).Member());

    // Checks that the operations are associative.
    CHECK(ApproxEqual(Plus(w1, Plus(w2, w3)), Plus(Plus(w1, w2), w3)));
    CHECK(ApproxEqual(Times(w1, Times(w2, w3)), Times(Times(w1, w2), w3)));

    // Checks the identity elements.
    CHECK(Plus(w1, Weight::Zero()) == w1);
    CHECK(Plus(Weight::Zero(), w1) == w1);
    CHECK(Times(w1, Weight::One()) == w1);
    CHECK(Times(Weight::One(), w1) == w1);

    // Check the no weight element.
    CHECK(!Weight::NoWeight().Member());
    CHECK(!Plus(w1, Weight::NoWeight()).Member());
    CHECK(!Plus(Weight::NoWeight(), w1).Member());
    CHECK(!Times(w1, Weight::NoWeight()).Member());
    CHECK(!Times(Weight::NoWeight(), w1).Member());

    // Checks that the operations commute.
    CHECK(ApproxEqual(Plus(w1, w2), Plus(w2, w1)));

    if (Weight::Properties() & kCommutative)
      CHECK(ApproxEqual(Times(w1, w2), Times(w2, w1)));

    // Checks Zero() is the annihilator.
    CHECK(Times(w1, Weight::Zero()) == Weight::Zero());
    CHECK(Times(Weight::Zero(), w1) == Weight::Zero());

    // Check Power(w, 0) is Weight::One()
    CHECK(Power(w1, 0) == Weight::One());

    // Check Power(w, 1) is w
    CHECK(Power(w1, 1) == w1);

    // Check Power(w, 3) is Times(w, Times(w, w))
    CHECK(Power(w1, 3) == Times(w1, Times(w1, w1)));

    // Checks distributivity.
    if (Weight::Properties() & kLeftSemiring) {
      CHECK(ApproxEqual(Times(w1, Plus(w2, w3)),
                        Plus(Times(w1, w2), Times(w1, w3))));
    }
    if (Weight::Properties() & kRightSemiring)
      CHECK(ApproxEqual(Times(Plus(w1, w2), w3),
                        Plus(Times(w1, w3), Times(w2, w3))));

    if (Weight::Properties() & kIdempotent) CHECK(Plus(w1, w1) == w1);

    if (Weight::Properties() & kPath)
      CHECK(Plus(w1, w2) == w1 || Plus(w1, w2) == w2);

    // Ensure weights form a left or right semiring.
    CHECK(Weight::Properties() & (kLeftSemiring | kRightSemiring));

    // Check when Times() is commutative that it is marked as a semiring.
    if (Weight::Properties() & kCommutative)
      CHECK(Weight::Properties() & kSemiring);
  }

  // Tests division operation.
  void TestDivision(Weight w1, Weight w2) {
    Weight p = Times(w1, w2);

    if (Weight::Properties() & kLeftSemiring) {
      Weight d = Divide(p, w1, DIVIDE_LEFT);
      if (d.Member()) CHECK(ApproxEqual(p, Times(w1, d)));
      CHECK(!Divide(w1, Weight::NoWeight(), DIVIDE_LEFT).Member());
      CHECK(!Divide(Weight::NoWeight(), w1, DIVIDE_LEFT).Member());
    }

    if (Weight::Properties() & kRightSemiring) {
      Weight d = Divide(p, w2, DIVIDE_RIGHT);
      if (d.Member()) CHECK(ApproxEqual(p, Times(d, w2)));
      CHECK(!Divide(w1, Weight::NoWeight(), DIVIDE_RIGHT).Member());
      CHECK(!Divide(Weight::NoWeight(), w1, DIVIDE_RIGHT).Member());
    }

    if (Weight::Properties() & kCommutative) {
      Weight d = Divide(p, w1, DIVIDE_RIGHT);
      if (d.Member()) CHECK(ApproxEqual(p, Times(d, w1)));
    }
  }

  // Tests reverse operation.
  void TestReverse(Weight w1, Weight w2) {
    typedef typename Weight::ReverseWeight ReverseWeight;

    ReverseWeight rw1 = w1.Reverse();
    ReverseWeight rw2 = w2.Reverse();

    CHECK(rw1.Reverse() == w1);
    CHECK(Plus(w1, w2).Reverse() == Plus(rw1, rw2));
    CHECK(Times(w1, w2).Reverse() == Times(rw2, rw1));
  }

  // Tests == is an equivalence relation.
  void TestEquality(Weight w1, Weight w2, Weight w3) {
    // Checks reflexivity.
    CHECK(w1 == w1);

    // Checks symmetry.
    CHECK((w1 == w2) == (w2 == w1));

    // Checks transitivity.
    if (w1 == w2 && w2 == w3) CHECK(w1 == w3);
  }

  // Tests binary serialization and textual I/O.
  void TestIO(Weight w) {
    // Tests binary I/O
    {
      std::ostringstream os;
      w.Write(os);
      os.flush();
      std::istringstream is(os.str());
      Weight v;
      v.Read(is);
      CHECK_EQ(w, v);
    }

    // Tests textual I/O.
    {
      std::ostringstream os;
      os << w;
      std::istringstream is(os.str());
      Weight v(Weight::One());
      is >> v;
      CHECK(ApproxEqual(w, v));
    }
  }

  // Tests copy constructor and assignment operator
  void TestCopy(Weight w) {
    Weight x = w;
    CHECK(w == x);

    x = Weight(w);
    CHECK(w == x);

    x.operator=(x);
    CHECK(w == x);
  }

  // Generates weights used in testing.
  WeightGenerator weight_generator_;
};

}  // namespace fst

#endif  // FST_TEST_WEIGHT_TESTER_H_
