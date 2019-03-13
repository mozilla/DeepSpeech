// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_EXTENSIONS_SPECIAL_PHI_FST_H_
#define FST_EXTENSIONS_SPECIAL_PHI_FST_H_

#include <memory>
#include <string>

#include <fst/const-fst.h>
#include <fst/matcher-fst.h>
#include <fst/matcher.h>

DECLARE_int64_t(phi_fst_phi_label);
DECLARE_bool(phi_fst_phi_loop);
DECLARE_string(phi_fst_rewrite_mode);

namespace fst {
namespace internal {

template <class Label>
class PhiFstMatcherData {
 public:
  PhiFstMatcherData(
      Label phi_label = FLAGS_phi_fst_phi_label,
      bool phi_loop = FLAGS_phi_fst_phi_loop,
      MatcherRewriteMode rewrite_mode = RewriteMode(FLAGS_phi_fst_rewrite_mode))
      : phi_label_(phi_label),
        phi_loop_(phi_loop),
        rewrite_mode_(rewrite_mode) {}

  PhiFstMatcherData(const PhiFstMatcherData &data)
      : phi_label_(data.phi_label_),
        phi_loop_(data.phi_loop_),
        rewrite_mode_(data.rewrite_mode_) {}

  static PhiFstMatcherData<Label> *Read(std::istream &istrm,
                                        const FstReadOptions &read) {
    auto *data = new PhiFstMatcherData<Label>();
    ReadType(istrm, &data->phi_label_);
    ReadType(istrm, &data->phi_loop_);
    int32_t rewrite_mode;
    ReadType(istrm, &rewrite_mode);
    data->rewrite_mode_ = static_cast<MatcherRewriteMode>(rewrite_mode);
    return data;
  }

  bool Write(std::ostream &ostrm, const FstWriteOptions &opts) const {
    WriteType(ostrm, phi_label_);
    WriteType(ostrm, phi_loop_);
    WriteType(ostrm, static_cast<int32_t>(rewrite_mode_));
    return !ostrm ? false : true;
  }

  Label PhiLabel() const { return phi_label_; }

  bool PhiLoop() const { return phi_loop_; }

  MatcherRewriteMode RewriteMode() const { return rewrite_mode_; }

 private:
  static MatcherRewriteMode RewriteMode(const string &mode) {
    if (mode == "auto") return MATCHER_REWRITE_AUTO;
    if (mode == "always") return MATCHER_REWRITE_ALWAYS;
    if (mode == "never") return MATCHER_REWRITE_NEVER;
    LOG(WARNING) << "PhiFst: Unknown rewrite mode: " << mode << ". "
                 << "Defaulting to auto.";
    return MATCHER_REWRITE_AUTO;
  }

  Label phi_label_;
  bool phi_loop_;
  MatcherRewriteMode rewrite_mode_;
};

}  // namespace internal

constexpr uint8_t kPhiFstMatchInput = 0x01;   // Input matcher is PhiMatcher.
constexpr uint8_t kPhiFstMatchOutput = 0x02;  // Output matcher is PhiMatcher.

template <class M, uint8_t flags = kPhiFstMatchInput | kPhiFstMatchOutput>
class PhiFstMatcher : public PhiMatcher<M> {
 public:
  using FST = typename M::FST;
  using Arc = typename M::Arc;
  using StateId = typename Arc::StateId;
  using Label = typename Arc::Label;
  using Weight = typename Arc::Weight;
  using MatcherData = internal::PhiFstMatcherData<Label>;

  enum : uint8_t { kFlags = flags };

  // This makes a copy of the FST.
  PhiFstMatcher(const FST &fst, MatchType match_type,
      std::shared_ptr<MatcherData> data = std::make_shared<MatcherData>())
      : PhiMatcher<M>(fst, match_type,
                      PhiLabel(match_type, data ? data->PhiLabel()
                                                : MatcherData().PhiLabel()),
                      data ? data->PhiLoop() : MatcherData().PhiLoop(),
                      data ? data->RewriteMode() : MatcherData().RewriteMode()),
        data_(data) {}

  // This doesn't copy the FST.
  PhiFstMatcher(const FST *fst, MatchType match_type,
      std::shared_ptr<MatcherData> data = std::make_shared<MatcherData>())
      : PhiMatcher<M>(fst, match_type,
                      PhiLabel(match_type, data ? data->PhiLabel()
                                                : MatcherData().PhiLabel()),
                      data ? data->PhiLoop() : MatcherData().PhiLoop(),
                      data ? data->RewriteMode() : MatcherData().RewriteMode()),
        data_(data) {}

  // This makes a copy of the FST.
  PhiFstMatcher(const PhiFstMatcher<M, flags> &matcher, bool safe = false)
      : PhiMatcher<M>(matcher, safe), data_(matcher.data_) {}

  PhiFstMatcher<M, flags> *Copy(bool safe = false) const override {
    return new PhiFstMatcher<M, flags>(*this, safe);
  }

  const MatcherData *GetData() const { return data_.get(); }

  std::shared_ptr<MatcherData> GetSharedData() const { return data_; }

 private:
  static Label PhiLabel(MatchType match_type, Label label) {
    if (match_type == MATCH_INPUT && flags & kPhiFstMatchInput) return label;
    if (match_type == MATCH_OUTPUT && flags & kPhiFstMatchOutput) return label;
    return kNoLabel;
  }

  std::shared_ptr<MatcherData> data_;
};

extern const char phi_fst_type[];
extern const char input_phi_fst_type[];
extern const char output_phi_fst_type[];

using StdPhiFst =
    MatcherFst<ConstFst<StdArc>, PhiFstMatcher<SortedMatcher<ConstFst<StdArc>>>,
               phi_fst_type>;

using LogPhiFst =
    MatcherFst<ConstFst<LogArc>, PhiFstMatcher<SortedMatcher<ConstFst<LogArc>>>,
               phi_fst_type>;

using Log64PhiFst = MatcherFst<ConstFst<Log64Arc>,
                               PhiFstMatcher<SortedMatcher<ConstFst<Log64Arc>>>,
                               input_phi_fst_type>;

using StdInputPhiFst =
    MatcherFst<ConstFst<StdArc>, PhiFstMatcher<SortedMatcher<ConstFst<StdArc>>,
                                               kPhiFstMatchInput>,
               input_phi_fst_type>;

using LogInputPhiFst =
    MatcherFst<ConstFst<LogArc>, PhiFstMatcher<SortedMatcher<ConstFst<LogArc>>,
                                               kPhiFstMatchInput>,
               input_phi_fst_type>;

using Log64InputPhiFst = MatcherFst<
    ConstFst<Log64Arc>,
    PhiFstMatcher<SortedMatcher<ConstFst<Log64Arc>>, kPhiFstMatchInput>,
    input_phi_fst_type>;

using StdOutputPhiFst =
    MatcherFst<ConstFst<StdArc>, PhiFstMatcher<SortedMatcher<ConstFst<StdArc>>,
                                               kPhiFstMatchOutput>,
               output_phi_fst_type>;

using LogOutputPhiFst =
    MatcherFst<ConstFst<LogArc>, PhiFstMatcher<SortedMatcher<ConstFst<LogArc>>,
                                               kPhiFstMatchOutput>,
               output_phi_fst_type>;

using Log64OutputPhiFst = MatcherFst<
    ConstFst<Log64Arc>,
    PhiFstMatcher<SortedMatcher<ConstFst<Log64Arc>>, kPhiFstMatchOutput>,
    output_phi_fst_type>;

}  // namespace fst

#endif  // FST_EXTENSIONS_SPECIAL_PHI_FST_H_
