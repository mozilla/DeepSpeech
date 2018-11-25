// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/extensions/pdt/getters.h>

namespace fst {
namespace script {

bool GetPdtComposeFilter(const string &str, PdtComposeFilter *cf) {
  if (str == "expand") {
    *cf = EXPAND_FILTER;
  } else if (str == "expand_paren") {
    *cf = EXPAND_PAREN_FILTER;
  } else if (str == "paren") {
    *cf = PAREN_FILTER;
  } else {
    return false;
  }
  return true;
}

bool GetPdtParserType(const string &str, PdtParserType *pt) {
  if (str == "left") {
    *pt = PDT_LEFT_PARSER;
  } else if (str == "left_sr") {
    *pt = PDT_LEFT_SR_PARSER;
  } else {
    return false;
  }
  return true;
}

}  // namespace script
}  // namespace fst
