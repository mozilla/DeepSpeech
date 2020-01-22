from __future__ import absolute_import, division, print_function

from . import swigwrapper # pylint: disable=import-self
from .swigwrapper import Alphabet

__version__ = swigwrapper.__version__

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
            serialized = alphabet.serialize()
            native_alphabet = swigwrapper.Alphabet()
            err = native_alphabet.deserialize(serialized, len(serialized))
            if err != 0:
                raise ValueError("Error when deserializing alphabet.")

            err = self.init(scorer_path.encode('utf-8'),
                            native_alphabet)
            if err != 0:
                raise ValueError("Scorer initialization failed with error code {}".format(err), err)

            self.reset_params(alpha, beta)

    def load_lm(self, lm_path):
        super(Scorer, self).load_lm(lm_path.encode('utf-8'))

    def save_dictionary(self, save_path, *args, **kwargs):
        super(Scorer, self).save_dictionary(save_path.encode('utf-8'), *args, **kwargs)


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
    serialized = alphabet.serialize()
    native_alphabet = swigwrapper.Alphabet()
    err = native_alphabet.deserialize(serialized, len(serialized))
    if err != 0:
        raise ValueError("Error when deserializing alphabet.")
    beam_results = swigwrapper.ctc_beam_search_decoder(
        probs_seq, native_alphabet, beam_size, cutoff_prob, cutoff_top_n,
        scorer)
    beam_results = [(res.confidence, alphabet.decode(res.tokens)) for res in beam_results]
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
    serialized = alphabet.serialize()
    native_alphabet = swigwrapper.Alphabet()
    err = native_alphabet.deserialize(serialized, len(serialized))
    if err != 0:
        raise ValueError("Error when deserializing alphabet.")
    batch_beam_results = swigwrapper.ctc_beam_search_decoder_batch(probs_seq, seq_lengths, native_alphabet, beam_size, num_processes, cutoff_prob, cutoff_top_n, scorer)
    batch_beam_results = [
        [(res.confidence, alphabet.decode(res.tokens)) for res in beam_results]
        for beam_results in batch_beam_results
    ]
    return batch_beam_results
