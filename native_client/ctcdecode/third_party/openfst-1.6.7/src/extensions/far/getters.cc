// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
//
// Definitions and functions for invoking and using Far main functions that
// support multiple and extensible arc types.

#include <fst/extensions/far/getters.h>

#include <string>
#include <vector>

#include <fstream>

namespace fst {

namespace script {

FarType GetFarType(const string &str) {
  if (str == "fst") {
    return FAR_FST;
  } else if (str == "stlist") {
    return FAR_STLIST;
  } else if (str == "sttable") {
    return FAR_STTABLE;
  } else {
    return FAR_DEFAULT;
  }
}

bool GetFarEntryType(const string &str, FarEntryType *entry_type) {
  if (str == "line") {
    *entry_type = FET_LINE;
  } else if (str == "file") {
    *entry_type = FET_FILE;
  } else {
    return false;
  }
  return true;
}

bool GetFarTokenType(const string &str, FarTokenType *token_type) {
  if (str == "symbol") {
    *token_type = FTT_SYMBOL;
  } else if (str == "byte") {
    *token_type = FTT_BYTE;
  } else if (str == "utf8") {
    *token_type = FTT_UTF8;
  } else {
    return false;
  }
  return true;
}

void ExpandArgs(int argc, char **argv, int *argcp, char ***argvp) {
}

}  // namespace script

string GetFarTypeString(FarType type) {
  switch (type) {
    case FAR_FST:
      return "fst";
    case FAR_STLIST:
      return "stlist";
    case FAR_STTABLE:
      return "sttable";
    case FAR_DEFAULT:
      return "default";
    default:
      return "<unknown>";
  }
}

}  // namespace fst
