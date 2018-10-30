// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// std::pair and std::tuple are used for the arguments of FstClass operations.
//
// If a function with a return value is required, use the WithReturnValue
// template as follows:
//
// WithReturnValue<bool, std::tuple<...>>

#ifndef FST_SCRIPT_ARG_PACKS_H_
#define FST_SCRIPT_ARG_PACKS_H_

#include <type_traits>

namespace fst {
namespace script {

// Tack this on to an existing type to add a return value. The syntax for
// accessing the args is then slightly more stilted, as you must do an extra
// member access (since the args are stored as a member of this class).

template <class Retval, class ArgTuple>
struct WithReturnValue {
  // Avoid reference-to-reference if ArgTuple is a reference.
  using Args = typename std::remove_reference<ArgTuple>::type;

  Retval retval;
  const Args &args;

  explicit WithReturnValue(const Args &args) : args(args) {}
};

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_ARG_PACKS_H_
