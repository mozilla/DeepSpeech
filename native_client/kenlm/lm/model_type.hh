#ifndef LM_MODEL_TYPE_H
#define LM_MODEL_TYPE_H

namespace lm {
namespace ngram {

/* Not the best numbering system, but it grew this way for historical reasons
 * and I want to preserve existing binary files. */
typedef enum {PROBING=0, REST_PROBING=1, TRIE=2, QUANT_TRIE=3, ARRAY_TRIE=4, QUANT_ARRAY_TRIE=5} ModelType;

// Historical names.
const ModelType HASH_PROBING = PROBING;
const ModelType TRIE_SORTED = TRIE;
const ModelType QUANT_TRIE_SORTED = QUANT_TRIE;
const ModelType ARRAY_TRIE_SORTED = ARRAY_TRIE;
const ModelType QUANT_ARRAY_TRIE_SORTED = QUANT_ARRAY_TRIE;

const static ModelType kQuantAdd = static_cast<ModelType>(QUANT_TRIE - TRIE);
const static ModelType kArrayAdd = static_cast<ModelType>(ARRAY_TRIE - TRIE);

} // namespace ngram
} // namespace lm
#endif // LM_MODEL_TYPE_H
