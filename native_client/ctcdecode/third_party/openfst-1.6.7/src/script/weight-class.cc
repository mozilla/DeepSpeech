// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/weight-class.h>

namespace fst {
namespace script {

REGISTER_FST_WEIGHT(StdArc::Weight);
REGISTER_FST_WEIGHT(LogArc::Weight);
REGISTER_FST_WEIGHT(Log64Arc::Weight);

WeightClass::WeightClass(const string &weight_type, const string &weight_str) {
  WeightClassRegister *reg = WeightClassRegister::GetRegister();
  StrToWeightImplBaseT stw = reg->GetEntry(weight_type);
  if (!stw) {
    FSTERROR() << "Unknown weight type: " << weight_type;
    impl_.reset();
    return;
  }
  impl_.reset(stw(weight_str, "WeightClass", 0));
}

WeightClass WeightClass::Zero(const string &weight_type) {
  return WeightClass(weight_type, __ZERO__);
}

WeightClass WeightClass::One(const string &weight_type) {
  return WeightClass(weight_type, __ONE__);
}

WeightClass WeightClass::NoWeight(const string &weight_type) {
  return WeightClass(weight_type, __NOWEIGHT__);
}

bool WeightClass::WeightTypesMatch(const WeightClass &other,
                                   const string &op_name) const {
  if (Type() != other.Type()) {
    FSTERROR() << "Weights with non-matching types passed to " << op_name
               << ": " << Type() << " and " << other.Type();
    return false;
  }
  return true;
}

bool operator==(const WeightClass &lhs, const WeightClass &rhs) {
  const auto *lhs_impl = lhs.GetImpl();
  const auto *rhs_impl = rhs.GetImpl();
  if (!(lhs_impl && rhs_impl && lhs.WeightTypesMatch(rhs, "operator=="))) {
    return false;
  }
  return *lhs_impl == *rhs_impl;
}

bool operator!=(const WeightClass &lhs, const WeightClass &rhs) {
  return !(lhs == rhs);
}

WeightClass Plus(const WeightClass &lhs, const WeightClass &rhs) {
  const auto *rhs_impl = rhs.GetImpl();
  if (!(lhs.GetImpl() && rhs_impl && lhs.WeightTypesMatch(rhs, "Plus"))) {
    return WeightClass();
  }
  WeightClass result(lhs);
  result.GetImpl()->PlusEq(*rhs_impl);
  return result;
}

WeightClass Times(const WeightClass &lhs, const WeightClass &rhs) {
  const auto *rhs_impl = rhs.GetImpl();
  if (!(lhs.GetImpl() && rhs_impl && lhs.WeightTypesMatch(rhs, "Plus"))) {
    return WeightClass();
  }
  WeightClass result(lhs);
  result.GetImpl()->TimesEq(*rhs_impl);
  return result;
}

WeightClass Divide(const WeightClass &lhs, const WeightClass &rhs) {
  const auto *rhs_impl = rhs.GetImpl();
  if (!(lhs.GetImpl() && rhs_impl && lhs.WeightTypesMatch(rhs, "Divide"))) {
    return WeightClass();
  }
  WeightClass result(lhs);
  result.GetImpl()->DivideEq(*rhs_impl);
  return result;
}

WeightClass Power(const WeightClass &weight, size_t n) {
  if (!weight.GetImpl()) return WeightClass();
  WeightClass result(weight);
  result.GetImpl()->PowerEq(n);
  return result;
}

std::ostream &operator<<(std::ostream &ostrm, const WeightClass &weight) {
  weight.impl_->Print(&ostrm);
  return ostrm;
}

}  // namespace script
}  // namespace fst
