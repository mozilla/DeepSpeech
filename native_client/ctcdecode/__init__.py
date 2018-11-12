from __future__ import absolute_import, division, print_function

from . import swigwrapper


class Scorer(swigwrapper.Scorer):
    """Wrapper for Scorer.

    :param alpha: Parameter associated with language model. Don't use
                  language model when alpha = 0.
    :type alpha: float
    :param beta: Parameter associated with word count. Don't use word
                 count when beta = 0.
    :type beta: float
    :model_path: Path to load language model.
    :trie_path: Path to trie file.
    :alphabet: Alphabet
    :type model_path: basestring
    """

    def __init__(self, alpha, beta, model_path, trie_path, alphabet):
        swigwrapper.Scorer.__init__(self, alpha, beta, model_path, trie_path, alphabet.config_file())


def ctc_beam_search_decoder(probs_seq,
                            alphabet,
                            beam_size,
                            cutoff_prob=1.0,
                            cutoff_top_n=40,
                            scorer=None):
    """Wrapper for the CTC Beam Search Decoder.

    :param probs_seq: 2-D list of probability distributions over each time
                      step, with each element being a list of normalized
                      probabilities over alphabet and blank.
    :type probs_seq: 2-D list
    :param alphabet: alphabet list.
    :alphabet: Alphabet
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param cutoff_prob: Cutoff probability in pruning,
                        default 1.0, no pruning.
    :type cutoff_prob: float
    :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n
                         characters with highest probs in alphabet will be
                         used in beam search, default 40.
    :type cutoff_top_n: int
    :param scorer: External scorer for partially decoded sentence, e.g. word
                   count or language model.
    :type scorer: Scorer
    :return: List of tuples of log probability and sentence as decoding
             results, in descending order of the probability.
    :rtype: list
    """
    beam_results = swigwrapper.ctc_beam_search_decoder(
        probs_seq, alphabet.config_file(), beam_size, cutoff_prob, cutoff_top_n,
        scorer)
    beam_results = [(res.probability, alphabet.decode(res.tokens)) for res in beam_results]
    return beam_results


def ctc_beam_search_decoder_batch(probs_seq,
                                  seq_lengths,
                                  alphabet,
                                  beam_size,
                                  num_processes,
                                  cutoff_prob=1.0,
                                  cutoff_top_n=40,
                                  scorer=None):
    """Wrapper for the batched CTC beam search decoder.

    :param probs_seq: 3-D list with each element as an instance of 2-D list
                      of probabilities used by ctc_beam_search_decoder().
    :type probs_seq: 3-D list
    :param alphabet: alphabet list.
    :alphabet: Alphabet
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param num_processes: Number of parallel processes.
    :type num_processes: int
    :param cutoff_prob: Cutoff probability in alphabet pruning,
                        default 1.0, no pruning.
    :type cutoff_prob: float
    :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n
                         characters with highest probs in alphabet will be
                         used in beam search, default 40.
    :type cutoff_top_n: int
    :param num_processes: Number of parallel processes.
    :type num_processes: int
    :param scorer: External scorer for partially decoded sentence, e.g. word
                   count or language model.
    :type scorer: Scorer
    :return: List of tuples of log probability and sentence as decoding
             results, in descending order of the probability.
    :rtype: list
    """
    batch_beam_results = swigwrapper.ctc_beam_search_decoder_batch(
        probs_seq, seq_lengths, alphabet.config_file(), beam_size, num_processes,
        cutoff_prob, cutoff_top_n, scorer)
    batch_beam_results = [
        [(res.probability, alphabet.decode(res.tokens)) for res in beam_results]
        for beam_results in batch_beam_results
    ]
    return batch_beam_results
