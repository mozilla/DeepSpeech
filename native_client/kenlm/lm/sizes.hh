#ifndef LM_SIZES_H
#define LM_SIZES_H

#include <vector>

#include <stdint.h>

namespace lm { namespace ngram {

struct Config;

void ShowSizes(const std::vector<uint64_t> &counts, const lm::ngram::Config &config);
void ShowSizes(const std::vector<uint64_t> &counts);
void ShowSizes(const char *file, const lm::ngram::Config &config);

}} // namespaces
#endif // LM_SIZES_H
