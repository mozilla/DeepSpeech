import os
import platform

#The API is not snake case which triggers linter errors
#pylint: disable=invalid-name

if platform.system().lower() == "windows":
    dslib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')

    # On Windows, we can't rely on RPATH being set to $ORIGIN/lib/ or on
    # @loader_path/lib
    if hasattr(os, 'add_dll_directory'):
        # Starting with Python 3.8 this properly handles the problem
        os.add_dll_directory(dslib_path)
    else:
        # Before Pythin 3.8 we need to change the PATH to include the proper
        # directory for the dynamic linker
        os.environ['PATH'] = dslib_path + ';' + os.environ['PATH']

import deepspeech

# rename for backwards compatibility
from deepspeech.impl import Version as version

class Model(object):
    """
    Class holding a DeepSpeech model

    :param aModelPath: Path to model file to load
    :type aModelPath: str
    """
    def __init__(self, model_path):
        # make sure the attribute is there if CreateModel fails
        self._impl = None

        status, impl = deepspeech.impl.CreateModel(model_path)
        if status != 0:
            raise RuntimeError("CreateModel failed with '{}' (0x{:X})".format(deepspeech.impl.ErrorCodeToErrorMessage(status),status))
        self._impl = impl

    def __del__(self):
        if self._impl:
            deepspeech.impl.FreeModel(self._impl)
            self._impl = None

    def beamWidth(self):
        """
        Get beam width value used by the model. If setModelBeamWidth was not
        called before, will return the default value loaded from the model file.

        :return: Beam width value used by the model.
        :type: int
        """
        return deepspeech.impl.GetModelBeamWidth(self._impl)

    def setBeamWidth(self, beam_width):
        """
        Set beam width value used by the model.

        :param beam_width: The beam width used by the model. A larger beam width value generates better results at the cost of decoding time.
        :type beam_width: int

        :return: Zero on success, non-zero on failure.
        :type: int
        """
        return deepspeech.impl.SetModelBeamWidth(self._impl, beam_width)

    def sampleRate(self):
        """
        Return the sample rate expected by the model.

        :return: Sample rate.
        :type: int
        """
        return deepspeech.impl.GetModelSampleRate(self._impl)

    def enableExternalScorer(self, scorer_path):
        """
        Enable decoding using an external scorer.

        :param scorer_path: The path to the external scorer file.
        :type scorer_path: str

        :throws: RuntimeError on error
        """
        status = deepspeech.impl.EnableExternalScorer(self._impl, scorer_path)
        if status != 0:
            raise RuntimeError("EnableExternalScorer failed with '{}' (0x{:X})".format(deepspeech.impl.ErrorCodeToErrorMessage(status),status))

    def disableExternalScorer(self):
        """
        Disable decoding using an external scorer.

        :return: Zero on success, non-zero on failure.
        """
        return deepspeech.impl.DisableExternalScorer(self._impl)

    def addHotWord(self, word, boost):
        """
        Add a word and its boost for decoding.

        :param word: the hot-word
        :type word: str

        :param word: the boost
        :type word: float

        :throws: RuntimeError on error
        """
        status = deepspeech.impl.AddHotWord(self._impl, word, boost)
        if status != 0:
            raise RuntimeError("AddHotWord failed with '{}' (0x{:X})".format(deepspeech.impl.ErrorCodeToErrorMessage(status),status))

    def eraseHotWord(self, word):
        """
        Remove entry for word from hot-words dict.

        :param word: the hot-word
        :type word: str

        :throws: RuntimeError on error
        """
        status = deepspeech.impl.EraseHotWord(self._impl, word)
        if status != 0:
            raise RuntimeError("EraseHotWord failed with '{}' (0x{:X})".format(deepspeech.impl.ErrorCodeToErrorMessage(status),status))

    def clearHotWords(self):
        """
        Remove all entries from hot-words dict.

        :throws: RuntimeError on error
        """
        status = deepspeech.impl.ClearHotWords(self._impl)
        if status != 0:
            raise RuntimeError("ClearHotWords failed with '{}' (0x{:X})".format(deepspeech.impl.ErrorCodeToErrorMessage(status),status))

    def setScorerAlphaBeta(self, alpha, beta):
        """
        Set hyperparameters alpha and beta of the external scorer.

        :param alpha: The alpha hyperparameter of the decoder. Language model weight.
        :type alpha: float

        :param beta: The beta hyperparameter of the decoder. Word insertion weight.
        :type beta: float

        :return: Zero on success, non-zero on failure.
        :type: int
        """
        return deepspeech.impl.SetScorerAlphaBeta(self._impl, alpha, beta)

    def stt(self, audio_buffer):
        """
        Use the DeepSpeech model to perform Speech-To-Text.

        :param audio_buffer: A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).
        :type audio_buffer: numpy.int16 array

        :return: The STT result.
        :type: str
        """
        return deepspeech.impl.SpeechToText(self._impl, audio_buffer)

    def sttWithMetadata(self, audio_buffer, num_results=1):
        """
        Use the DeepSpeech model to perform Speech-To-Text and return results including metadata.

        :param audio_buffer: A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).
        :type audio_buffer: numpy.int16 array

        :param num_results: Maximum number of candidate transcripts to return. Returned list might be smaller than this.
        :type num_results: int

        :return: Metadata object containing multiple candidate transcripts. Each transcript has per-token metadata including timing information.
        :type: :func:`Metadata`
        """
        return deepspeech.impl.SpeechToTextWithMetadata(self._impl, audio_buffer, num_results)

    def createStream(self):
        """
        Create a new streaming inference state. The streaming state returned by
        this function can then be passed to :func:`feedAudioContent()` and :func:`finishStream()`.

        :return: Stream object representing the newly created stream
        :type: :func:`Stream`

        :throws: RuntimeError on error
        """
        status, ctx = deepspeech.impl.CreateStream(self._impl)
        if status != 0:
            raise RuntimeError("CreateStream failed with '{}' (0x{:X})".format(deepspeech.impl.ErrorCodeToErrorMessage(status),status))
        return Stream(ctx)


class Stream(object):
    """
    Class wrapping a DeepSpeech stream. The constructor cannot be called directly.
    Use :func:`Model.createStream()`
    """
    def __init__(self, native_stream):
        self._impl = native_stream

    def __del__(self):
        if self._impl:
            self.freeStream()

    def feedAudioContent(self, audio_buffer):
        """
        Feed audio samples to an ongoing streaming inference.

        :param audio_buffer: A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).
        :type audio_buffer: numpy.int16 array

        :throws: RuntimeError if the stream object is not valid
        """
        if not self._impl:
            raise RuntimeError("Stream object is not valid. Trying to feed an already finished stream?")
        deepspeech.impl.FeedAudioContent(self._impl, audio_buffer)

    def intermediateDecode(self):
        """
        Compute the intermediate decoding of an ongoing streaming inference.

        :return: The STT intermediate result.
        :type: str

        :throws: RuntimeError if the stream object is not valid
        """
        if not self._impl:
            raise RuntimeError("Stream object is not valid. Trying to decode an already finished stream?")
        return deepspeech.impl.IntermediateDecode(self._impl)

    def intermediateDecodeWithMetadata(self, num_results=1):
        """
        Compute the intermediate decoding of an ongoing streaming inference and return results including metadata.

        :param num_results: Maximum number of candidate transcripts to return. Returned list might be smaller than this.
        :type num_results: int

        :return: Metadata object containing multiple candidate transcripts. Each transcript has per-token metadata including timing information.
        :type: :func:`Metadata`

        :throws: RuntimeError if the stream object is not valid
        """
        if not self._impl:
            raise RuntimeError("Stream object is not valid. Trying to decode an already finished stream?")
        return deepspeech.impl.IntermediateDecodeWithMetadata(self._impl, num_results)

    def finishStream(self):
        """
        Compute the final decoding of an ongoing streaming inference and return
        the result. Signals the end of an ongoing streaming inference. The underlying
        stream object must not be used after this method is called.

        :return: The STT result.
        :type: str

        :throws: RuntimeError if the stream object is not valid
        """
        if not self._impl:
            raise RuntimeError("Stream object is not valid. Trying to finish an already finished stream?")
        result = deepspeech.impl.FinishStream(self._impl)
        self._impl = None
        return result

    def finishStreamWithMetadata(self, num_results=1):
        """
        Compute the final decoding of an ongoing streaming inference and return
        results including metadata. Signals the end of an ongoing streaming
        inference. The underlying stream object must not be used after this
        method is called.

        :param num_results: Maximum number of candidate transcripts to return. Returned list might be smaller than this.
        :type num_results: int

        :return: Metadata object containing multiple candidate transcripts. Each transcript has per-token metadata including timing information.
        :type: :func:`Metadata`

        :throws: RuntimeError if the stream object is not valid
        """
        if not self._impl:
            raise RuntimeError("Stream object is not valid. Trying to finish an already finished stream?")
        result = deepspeech.impl.FinishStreamWithMetadata(self._impl, num_results)
        self._impl = None
        return result

    def freeStream(self):
        """
        Destroy a streaming state without decoding the computed logits. This can
        be used if you no longer need the result of an ongoing streaming inference.

        :throws: RuntimeError if the stream object is not valid
        """
        if not self._impl:
            raise RuntimeError("Stream object is not valid. Trying to free an already finished stream?")
        deepspeech.impl.FreeStream(self._impl)
        self._impl = None


# This is only for documentation purpose
# Metadata, CandidateTranscript and TokenMetadata should be in sync with native_client/deepspeech.h
class TokenMetadata(object):
    """
    Stores each individual character, along with its timing information
    """

    def text(self):
        """
        The text for this token
        """


    def timestep(self):
        """
        Position of the token in units of 20ms
        """


    def start_time(self):
        """
        Position of the token in seconds
        """


class CandidateTranscript(object):
    """
    Stores the entire CTC output as an array of character metadata objects
    """
    def tokens(self):
        """
        List of tokens

        :return: A list of :func:`TokenMetadata` elements
        :type: list
        """


    def confidence(self):
        """
        Approximated confidence value for this transcription. This is roughly the
        sum of the acoustic model logit values for each timestep/character that
        contributed to the creation of this transcription.
        """


class Metadata(object):
    def transcripts(self):
        """
        List of candidate transcripts

        :return: A list of :func:`CandidateTranscript` objects
        :type: list
        """
