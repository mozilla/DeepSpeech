#ifndef CTC_BEAM_SEARCH_DECODER_H_
#define CTC_BEAM_SEARCH_DECODER_H_

#include <string>
#include <vector>

#include "scorer.h"
#include "output.h"
#include "alphabet.h"
#include "decoderstate.h"

/* Initialize CTC beam search decoder

 * Parameters:
 *     alphabet: The alphabet.
 *     class_dim: Alphabet length (plus 1 for space character).
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A struct containing prefixes and state variables. 
*/
DecoderState* decoder_init(const Alphabet &alphabet,
                           int class_dim,
                           Scorer *ext_scorer);

/* Send data to the decoder

 * Parameters:
 *     probs: 2-D vector where each element is a vector of probabilities
 *               over alphabet of one time step. 
 *     alphabet: The alphabet.
 *     state: The state structure previously obtained from decoder_init().
 *     time_dim: Number of timesteps.
 *     class_dim: Alphabet length (plus 1 for space character).
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     beam_size: The width of beam search.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
*/
void decoder_next(const double *probs,
                  const Alphabet &alphabet,
                  DecoderState *state,
                  int time_dim,
                  int class_dim,
                  double cutoff_prob,
                  size_t cutoff_top_n,
                  size_t beam_size,
                  Scorer *ext_scorer);

/* Get transcription for the data you sent via decoder_next()

 * Parameters:
 *     state: The state structure previously obtained from decoder_init().
 *     alphabet: The alphabet.
 *     beam_size: The width of beam search.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A vector where each element is a pair of score and decoding result,
 *     in descending order.
*/
std::vector<Output> decoder_decode(DecoderState *state,
                                   const Alphabet &alphabet,
                                   size_t beam_size,
                                   Scorer* ext_scorer);

/* CTC Beam Search Decoder
 * Parameters:
 *     probs: 2-D vector where each element is a vector of probabilities
 *            over alphabet of one time step.
 *     time_dim: Number of timesteps.
 *     class_dim: Alphabet length (plus 1 for space character).
 *     alphabet: The alphabet.
 *     beam_size: The width of beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A vector where each element is a pair of score and decoding result,
 *     in descending order.
*/

std::vector<Output> ctc_beam_search_decoder(
    const double* probs,
    int time_dim,
    int class_dim,
    const Alphabet &alphabet,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer);

/* CTC Beam Search Decoder for batch data
 * Parameters:
 *     probs: 3-D vector where each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *     alphabet: The alphabet.
 *     beam_size: The width of beam search.
 *     num_processes: Number of threads for beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A 2-D vector where each element is a vector of beam search decoding
 *     result for one audio sample.
*/
std::vector<std::vector<Output>>
ctc_beam_search_decoder_batch(
    const double* probs,
    int batch_size,
    int time_dim,
    int class_dim,
    const int* seq_lengths,
    int seq_lengths_size,
    const Alphabet &alphabet,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer);

#endif  // CTC_BEAM_SEARCH_DECODER_H_
