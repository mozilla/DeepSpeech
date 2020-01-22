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
from deepspeech.impl import PrintVersions as printVersions

class Model(object):
    """
    Class holding a DeepSpeech model

    :param aModelPath: Path to model file to load
    :type aModelPath: str

    :param aBeamWidth: Decoder beam width
    :type aBeamWidth: int
    """
    def __init__(self,  *args, **kwargs):
        # make sure the attribute is there if CreateModel fails
        self._impl = None

        status, impl = deepspeech.impl.CreateModel(*args, **kwargs)
        if status != 0:
            raise RuntimeError("CreateModel failed with error code {}".format(status))
        self._impl = impl

    def __del__(self):
        if self._impl:
            deepspeech.impl.FreeModel(self._impl)
            self._impl = None

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

        :return: Zero on success, non-zero on failure.
        :type: int
        """
        return deepspeech.impl.EnableExternalScorer(self._impl, scorer_path)

    def disableExternalScorer(self):
        """
        Disable decoding using an external scorer.

        :return: Zero on success, non-zero on failure.
        """
        return deepspeech.impl.DisableExternalScorer(self._impl)

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

    def sttWithMetadata(self, audio_buffer):
        """
        Use the DeepSpeech model to perform Speech-To-Text and output metadata about the results.

        :param audio_buffer: A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).
        :type audio_buffer: numpy.int16 array

        :return: Outputs a struct of individual letters along with their timing information.
        :type: :func:`Metadata`
        """
        return deepspeech.impl.SpeechToTextWithMetadata(self._impl, audio_buffer)

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
            raise RuntimeError("CreateStream failed with error code {}".format(status))
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

    def finishStream(self):
        """
        Signal the end of an audio signal to an ongoing streaming inference,
        returns the STT result over the whole audio signal.

        :return: The STT result.
        :type: str

        :throws: RuntimeError if the stream object is not valid
        """
        if not self._impl:
            raise RuntimeError("Stream object is not valid. Trying to finish an already finished stream?")
        result = deepspeech.impl.FinishStream(self._impl)
        self._impl = None
        return result

    def finishStreamWithMetadata(self):
        """
        Signal the end of an audio signal to an ongoing streaming inference,
        returns per-letter metadata.

        :return: Outputs a struct of individual letters along with their timing information.
        :type: :func:`Metadata`

        :throws: RuntimeError if the stream object is not valid
        """
        if not self._impl:
            raise RuntimeError("Stream object is not valid. Trying to finish an already finished stream?")
        result = deepspeech.impl.FinishStreamWithMetadata(self._impl)
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
# Metadata and MetadataItem should be in sync with native_client/deepspeech.h
class MetadataItem(object):
    """
    Stores each individual character, along with its timing information
    """

    def character(self):
        """
        The character generated for transcription
        """


    def timestep(self):
        """
        Position of the character in units of 20ms
        """


    def start_time(self):
        """
        Position of the character in seconds
        """


class Metadata(object):
    """
    Stores the entire CTC output as an array of character metadata objects
    """
    def items(self):
        """
        List of items

        :return: A list of :func:`MetadataItem` elements
        :type: list
        """


    def num_items(self):
        """
        Size of the list of items

        :return: Size of the list of items
        :type: int
        """


    def confidence(self):
        """
        Approximated confidence value for this transcription. This is roughly the
        sum of the acoustic model logit values for each timestep/character that
        contributed to the creation of this transcription.
        """

