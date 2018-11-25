// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Utilities for reading and writing textual strings representing states,
// labels, and weights and files specifying label-label pairs and potentials
// (state-weight pairs).

#ifndef FST_SCRIPT_TEXT_IO_H__
#define FST_SCRIPT_TEXT_IO_H__

#include <string>
#include <vector>

#include <fst/script/weight-class.h>

namespace fst {
namespace script {

bool ReadPotentials(const string &weight_type, const string &filename,
                    std::vector<WeightClass> *potentials);

bool WritePotentials(const string &filename,
                     const std::vector<WeightClass> &potentials);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_TEXT_IO_H__
