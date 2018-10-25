#ifndef CTC_BEAM_SEARCH_DECODER_H_
#define CTC_BEAM_SEARCH_DECODER_H_

#include <string>
#include <utility>
#include <vector>

#include "scorer.h"
#include "output.h"
#include "alphabet.h"

/* CTC Beam Search Decoder

 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     vocabulary: A vector of vocabulary.
 *     beam_size: The width of beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A vector that each element is a pair of score  and decoding result,
 *     in desending order.
*/

std::vector<std::pair<double, Output>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const Alphabet &vocabulary,
    size_t beam_size,
    double cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    Scorer *ext_scorer = nullptr);

/* CTC Beam Search Decoder for batch data

 * Parameters:
 *     probs_seq: 3-D vector that each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *     vocabulary: A vector of vocabulary.
 *     beam_size: The width of beam search.
 *     num_processes: Number of threads for beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A 2-D vector that each element is a vector of beam search decoding
 *     result for one audio sample.
*/
std::vector<std::vector<std::pair<double, Output>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    size_t blank_id = 0,
    Scorer *ext_scorer = nullptr);

#endif  // CTC_BEAM_SEARCH_DECODER_H_
