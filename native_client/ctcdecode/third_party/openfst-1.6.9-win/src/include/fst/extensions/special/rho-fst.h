// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_EXTENSIONS_SPECIAL_RHO_FST_H_
#define FST_EXTENSIONS_SPECIAL_RHO_FST_H_

#include <memory>
#include <string>

#include <fst/const-fst.h>
#include <fst/matcher-fst.h>
#include <fst/matcher.h>

DECLARE_int64_t(rho_fst_rho_label);
DECLARE_string(rho_fst_rewrite_mode);

namespace fst {
namespace internal {

template <class Label>
class RhoFstMatcherData {
 public:
  explicit RhoFstMatcherData(
      Label rho_label = FLAGS_rho_fst_rho_label,
      MatcherRewriteMode rewrite_mode = RewriteMode(FLAGS_rho_fst_rewrite_mode))
      : rho_label_(rho_label), rewrite_mode_(rewrite_mode) {}

  RhoFstMatcherData(const RhoFstMatcherData &data)
      : rho_label_(data.rho_label_), rewrite_mode_(data.rewrite_mode_) {}

  static RhoFstMatcherData<Label> *Read(std::istream &istrm,
                                    const FstReadOptions &read) {
    auto *data = new RhoFstMatcherData<Label>();
    ReadType(istrm, &data->rho_label_);
    int32_t rewrite_mode;
    ReadType(istrm, &rewrite_mode);
    data->rewrite_mode_ = static_cast<MatcherRewriteMode>(rewrite_mode);
    return data;
  }

  bool Write(std::ostream &ostrm, const FstWriteOptions &opts) const {
    WriteType(ostrm, rho_label_);
    WriteType(ostrm, static_cast<int32_t>(rewrite_mode_));
    return !ostrm ? false : true;
  }

  Label RhoLabel() const { return rho_label_; }

  MatcherRewriteMode RewriteMode() const { return rewrite_mode_; }

 private:
  static MatcherRewriteMode RewriteMode(const string &mode) {
    if (mode == "auto") return MATCHER_REWRITE_AUTO;
    if (mode == "always") return MATCHER_REWRITE_ALWAYS;
    if (mode == "never") return MATCHER_REWRITE_NEVER;
    LOG(WARNING) << "RhoFst: Unknown rewrite mode: " << mode << ". "
                 << "Defaulting to auto.";
    return MATCHER_REWRITE_AUTO;
  }

  Label rho_label_;
  MatcherRewriteMode rewrite_mode_;
};

}  // namespace internal

constexpr uint8_t kRhoFstMatchInput = 0x01;   // Input matcher is RhoMatcher.
constexpr uint8_t kRhoFstMatchOutput = 0x02;  // Output matcher is RhoMatcher.

template <class M, uint8_t flags = kRhoFstMatchInput | kRhoFstMatchOutput>
class RhoFstMatcher : public RhoMatcher<M> {
 public:
  using FST = typename M::FST;
  using Arc = typename M::Arc;
  using StateId = typename Arc::StateId;
  using Label = typename Arc::Label;
  using Weight = typename Arc::Weight;
  using MatcherData = internal::RhoFstMatcherData<Label>;

  enum : uint8_t { kFlags = flags };

  // This makes a copy of the FST.
  RhoFstMatcher(
      const FST &fst, MatchType match_type,
      std::shared_ptr<MatcherData> data = std::make_shared<MatcherData>())
      : RhoMatcher<M>(fst, match_type,
                      RhoLabel(match_type, data ? data->RhoLabel()
                                                : MatcherData().RhoLabel()),
                      data ? data->RewriteMode() : MatcherData().RewriteMode()),
        data_(data) {}

  // This doesn't copy the FST.
  RhoFstMatcher(
      const FST *fst, MatchType match_type,
      std::shared_ptr<MatcherData> data = std::make_shared<MatcherData>())
      : RhoMatcher<M>(fst, match_type,
                      RhoLabel(match_type, data ? data->RhoLabel()
                                                : MatcherData().RhoLabel()),
                      data ? data->RewriteMode() : MatcherData().RewriteMode()),
        data_(data) {}

  // This makes a copy of the FST.
  RhoFstMatcher(const RhoFstMatcher<M, flags> &matcher, bool safe = false)
      : RhoMatcher<M>(matcher, safe), data_(matcher.data_) {}

  RhoFstMatcher<M, flags> *Copy(bool safe = false) const override {
    return new RhoFstMatcher<M, flags>(*this, safe);
  }

  const MatcherData *GetData() const { return data_.get(); }

  std::shared_ptr<MatcherData> GetSharedData() const { return data_; }

 private:
  static Label RhoLabel(MatchType match_type, Label label) {
    if (match_type == MATCH_INPUT && flags & kRhoFstMatchInput) return label;
    if (match_type == MATCH_OUTPUT && flags & kRhoFstMatchOutput) return label;
    return kNoLabel;
  }

  std::shared_ptr<MatcherData> data_;
};

extern const char rho_fst_type[];
extern const char input_rho_fst_type[];
extern const char output_rho_fst_type[];

using StdRhoFst =
    MatcherFst<ConstFst<StdArc>, RhoFstMatcher<SortedMatcher<ConstFst<StdArc>>>,
               rho_fst_type>;

using LogRhoFst =
    MatcherFst<ConstFst<LogArc>, RhoFstMatcher<SortedMatcher<ConstFst<LogArc>>>,
               rho_fst_type>;

using Log64RhoFst = MatcherFst<ConstFst<Log64Arc>,
                               RhoFstMatcher<SortedMatcher<ConstFst<Log64Arc>>>,
                               input_rho_fst_type>;

using StdInputRhoFst =
    MatcherFst<ConstFst<StdArc>, RhoFstMatcher<SortedMatcher<ConstFst<StdArc>>,
                                               kRhoFstMatchInput>,
               input_rho_fst_type>;

using LogInputRhoFst =
    MatcherFst<ConstFst<LogArc>, RhoFstMatcher<SortedMatcher<ConstFst<LogArc>>,
                                               kRhoFstMatchInput>,
               input_rho_fst_type>;

using Log64InputRhoFst = MatcherFst<
    ConstFst<Log64Arc>,
    RhoFstMatcher<SortedMatcher<ConstFst<Log64Arc>>, kRhoFstMatchInput>,
    input_rho_fst_type>;

using StdOutputRhoFst =
    MatcherFst<ConstFst<StdArc>, RhoFstMatcher<SortedMatcher<ConstFst<StdArc>>,
                                               kRhoFstMatchOutput>,
               output_rho_fst_type>;

using LogOutputRhoFst =
    MatcherFst<ConstFst<LogArc>, RhoFstMatcher<SortedMatcher<ConstFst<LogArc>>,
                                               kRhoFstMatchOutput>,
               output_rho_fst_type>;

using Log64OutputRhoFst = MatcherFst<
    ConstFst<Log64Arc>,
    RhoFstMatcher<SortedMatcher<ConstFst<Log64Arc>>, kRhoFstMatchOutput>,
    output_rho_fst_type>;

}  // namespace fst

#endif  // FST_EXTENSIONS_SPECIAL_RHO_FST_H_
