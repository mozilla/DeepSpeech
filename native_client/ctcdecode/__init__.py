from __future__ import absolute_import, division, print_function

from . import swigwrapper # pylint: disable=import-self
from .swigwrapper import UTF8Alphabet

__version__ = swigwrapper.__version__

# Hack: import error codes by matching on their names, as SWIG unfortunately
# does not support binding enums to Python in a scoped manner yet.
for symbol in dir(swigwrapper):
    if symbol.startswith('DS_ERR_'):
        globals()[symbol] = getattr(swigwrapper, symbol)

class Scorer(swigwrapper.Scorer):
    """Wrapper for Scorer.

    :param alpha: Language model weight.
    :type alpha: float
    :param beta: Word insertion bonus.
    :type beta: float
    :scorer_path: Path to load scorer from.
    :alphabet: Alphabet
    :type scorer_path: basestring
    """
    def __init__(self, alpha=None, beta=None, scorer_path=None, alphabet=None):
        super(Scorer, self).__init__()
        # Allow bare initialization
        if alphabet:
            assert alpha is not None, 'alpha parameter is required'
            assert beta is not None, 'beta parameter is required'
            assert scorer_path, 'scorer_path parameter is required'

            err = self.init(scorer_path, alphabet)
            if err != 0:
                raise ValueError('Scorer initialization failed with error code 0x{:X}'.format(err))

            self.reset_params(alpha, beta)


class Alphabet(swigwrapper.Alphabet):
    """Convenience wrapper for Alphabet which calls init in the constructor"""
    def __init__(self, config_path):
        super(Alphabet, self).__init__()
        err = self.init(config_path)
        if err != 0:
            raise ValueError('Alphabet initialization failed with error code 0x{:X}'.format(err))


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
    :param alphabet: Alphabet
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
    :return: List of tuples of confidence and sentence as decoding
             results, in descending order of the confidence.
    :rtype: list
    """
    beam_results = swigwrapper.ctc_beam_search_decoder(
        probs_seq, alphabet, beam_size, cutoff_prob, cutoff_top_n,
        scorer)
    beam_results = [(res.confidence, alphabet.Decode(res.tokens)) for res in beam_results]
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
    :return: List of tuples of confidence and sentence as decoding
             results, in descending order of the confidence.
    :rtype: list
    """
    batch_beam_results = swigwrapper.ctc_beam_search_decoder_batch(probs_seq, seq_lengths, alphabet, beam_size, num_processes, cutoff_prob, cutoff_top_n, scorer)
    batch_beam_results = [
        [(res.confidence, alphabet.Decode(res.tokens)) for res in beam_results]
        for beam_results in batch_beam_results
    ]
    return batch_beam_results
