/* Map vocab ids.  This is useful to merge independently collected counts or
 * change the vocab ids to the order used by the trie.
 */
#ifndef LM_COMMON_RENUMBER_H
#define LM_COMMON_RENUMBER_H

#include "lm/word_index.hh"

#include <cstddef>

namespace util { namespace stream { class ChainPosition; }}

namespace lm {

class Renumber {
  public:
    // Assumes the array is large enough to map all words and stays alive while
    // the thread is active.
    Renumber(const WordIndex *new_numbers, std::size_t order)
      : new_numbers_(new_numbers), order_(order) {}

    void Run(const util::stream::ChainPosition &position);

  private:
    const WordIndex *new_numbers_;
    std::size_t order_;
};

} // namespace lm
#endif // LM_COMMON_RENUMBER_H
