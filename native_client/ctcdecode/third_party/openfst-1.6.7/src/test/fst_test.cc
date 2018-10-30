// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Regression test for FST classes.

#include "./fst_test.h"

#include <utility>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/compact-fst.h>
#include <fst/const-fst.h>
#include <fst/edit-fst.h>
#include <fst/matcher-fst.h>

namespace fst {
namespace {

// A user-defined arc type.
struct CustomArc {
  typedef int16 Label;
  typedef ProductWeight<TropicalWeight, LogWeight> Weight;
  typedef int64 StateId;

  CustomArc(Label i, Label o, Weight w, StateId s)
      : ilabel(i), olabel(o), weight(std::move(w)), nextstate(s) {}
  CustomArc() {}

  static const string &Type() {  // Arc type name
    static const string *const type = new string("my");
    return *type;
  }

  Label ilabel;       // Transition input label
  Label olabel;       // Transition output label
  Weight weight;      // Transition weight
  StateId nextstate;  // Transition destination state
};

// A user-defined compactor for test FST.
template <class A>
class CustomCompactor {
 public:
  typedef A Arc;
  typedef typename A::Label Label;
  typedef typename A::StateId StateId;
  typedef typename A::Weight Weight;
  typedef std::pair<Label, Weight> Element;

  Element Compact(StateId s, const A &arc) const {
    return std::make_pair(arc.ilabel, arc.weight);
  }

  Arc Expand(StateId s, const Element &p, uint32 f = kArcValueFlags) const {
    return p.first == kNoLabel ? Arc(kNoLabel, kNoLabel, p.second, kNoStateId)
                               : Arc(p.first, 0, p.second, s);
  }

  ssize_t Size() const { return -1; }

  uint64 Properties() const { return 0ULL; }

  bool Compatible(const Fst<A> &fst) const { return true; }

  static const string &Type() {
    static const string *const type = new string("my");
    return *type;
  }

  bool Write(std::ostream &strm) const { return true; }

  static CustomCompactor *Read(std::istream &strm) {
    return new CustomCompactor;
  }
};

REGISTER_FST(VectorFst, CustomArc);
REGISTER_FST(ConstFst, CustomArc);
static fst::FstRegisterer<CompactFst<StdArc, CustomCompactor<StdArc>>>
    CompactFst_StdArc_CustomCompactor_registerer;
static fst::FstRegisterer<CompactFst<CustomArc, CustomCompactor<CustomArc>>>
    CompactFst_CustomArc_CustomCompactor_registerer;
static fst::FstRegisterer<ConstFst<StdArc, uint16>>
    ConstFst_StdArc_uint16_registerer;
static fst::FstRegisterer<
    CompactFst<StdArc, CustomCompactor<StdArc>, uint16>>
    CompactFst_StdArc_CustomCompactor_uint16_registerer;

}  // namespace
}  // namespace fst

using fst::FstTester;
using fst::VectorFst;
using fst::ConstFst;
using fst::MatcherFst;
using fst::CompactFst;
using fst::Fst;
using fst::StdArc;
using fst::CustomArc;
using fst::CustomCompactor;
using fst::StdArcLookAheadFst;
using fst::EditFst;

int main(int argc, char **argv) {
  FLAGS_fst_verify_properties = true;
  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(argv[0], &argc, &argv, true);

  // VectorFst<StdArc> tests
  {
    FstTester<VectorFst<StdArc>> std_vector_tester;
    std_vector_tester.TestBase();
    std_vector_tester.TestExpanded();
    std_vector_tester.TestAssign();
    std_vector_tester.TestCopy();
    std_vector_tester.TestIO();
    std_vector_tester.TestMutable();
  }

  // ConstFst<StdArc> tests
  {
    FstTester<ConstFst<StdArc>> std_const_tester;
    std_const_tester.TestBase();
    std_const_tester.TestExpanded();
    std_const_tester.TestCopy();
    std_const_tester.TestIO();
  }

  // CompactFst<StdArc, CustomCompactor<StdArc>>
  {
    FstTester<CompactFst<StdArc, CustomCompactor<StdArc>>> std_compact_tester;
    std_compact_tester.TestBase();
    std_compact_tester.TestExpanded();
    std_compact_tester.TestCopy();
    std_compact_tester.TestIO();
  }

  // VectorFst<CustomArc> tests
  {
    FstTester<VectorFst<CustomArc>> std_vector_tester;
    std_vector_tester.TestBase();
    std_vector_tester.TestExpanded();
    std_vector_tester.TestAssign();
    std_vector_tester.TestCopy();
    std_vector_tester.TestIO();
    std_vector_tester.TestMutable();
  }

  // ConstFst<CustomArc> tests
  {
    FstTester<ConstFst<CustomArc>> std_const_tester;
    std_const_tester.TestBase();
    std_const_tester.TestExpanded();
    std_const_tester.TestCopy();
    std_const_tester.TestIO();
  }

  // CompactFst<CustomArc, CustomCompactor<CustomArc>>
  {
    FstTester<CompactFst<CustomArc, CustomCompactor<CustomArc>>>
        std_compact_tester;
    std_compact_tester.TestBase();
    std_compact_tester.TestExpanded();
    std_compact_tester.TestCopy();
    std_compact_tester.TestIO();
  }

  // ConstFst<StdArc, uint16> tests
  {
    FstTester<ConstFst<StdArc, uint16>> std_const_tester;
    std_const_tester.TestBase();
    std_const_tester.TestExpanded();
    std_const_tester.TestCopy();
    std_const_tester.TestIO();
  }

  // CompactFst<StdArc, CustomCompactor<StdArc>, uint16>
  {
    FstTester<CompactFst<StdArc, CustomCompactor<StdArc>, uint16>>
        std_compact_tester;
    std_compact_tester.TestBase();
    std_compact_tester.TestExpanded();
    std_compact_tester.TestCopy();
    std_compact_tester.TestIO();
  }

  // FstTester<StdArcLookAheadFst>
  {
    FstTester<StdArcLookAheadFst> std_matcher_tester;
    std_matcher_tester.TestBase();
    std_matcher_tester.TestExpanded();
    std_matcher_tester.TestCopy();
  }

  // EditFst<StdArc> tests
  {
    FstTester<EditFst<StdArc>> std_edit_tester;
    std_edit_tester.TestBase();
    std_edit_tester.TestExpanded();
    std_edit_tester.TestAssign();
    std_edit_tester.TestCopy();
    std_edit_tester.TestMutable();
  }

  std::cout << "PASS" << std::endl;

  return 0;
}
