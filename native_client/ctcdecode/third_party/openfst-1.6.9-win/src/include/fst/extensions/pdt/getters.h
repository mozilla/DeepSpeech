// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_EXTENSIONS_PDT_GETTERS_H_
#define FST_EXTENSIONS_PDT_GETTERS_H_

#include <string>

#include <fst/extensions/pdt/compose.h>
#include <fst/extensions/pdt/replace.h>

namespace fst {
namespace script {

bool GetPdtComposeFilter(const string &str, PdtComposeFilter *cf);

bool GetPdtParserType(const string &str, PdtParserType *pt);

}  // namespace script
}  // namespace fst

#endif  // FST_EXTENSIONS_PDT_GETTERS_H_
