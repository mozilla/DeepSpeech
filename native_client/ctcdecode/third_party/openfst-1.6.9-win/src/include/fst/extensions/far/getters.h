// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes and functions for registering and invoking FAR main
// functions that support multiple and extensible arc types.

#ifndef FST_EXTENSIONS_FAR_GETTERS_H_
#define FST_EXTENSIONS_FAR_GETTERS_H_

#include <fst/flags.h>
#include <fst/extensions/far/far.h>

namespace fst {
namespace script {

FarType GetFarType(const string &str);

bool GetFarEntryType(const string &str, FarEntryType *entry_type);

bool GetFarTokenType(const string &str, FarTokenType *token_type);

void ExpandArgs(int argc, char **argv, int *argcp, char ***argvp);

}  // namespace script

string GetFarTypeString(FarType type);

}  // namespace fst

#endif  // FST_EXTENSIONS_FAR_GETTERS_H_
