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
from deepspeech.impl import FreeStream as freeStream

class Model(object):
    """
    Class holding a DeepSpeech model

    :param aModelPath: Path to model file to load
    :type aModelPath: str

    :param aAlphabetConfigPath: Path to alphabet file to load
    :type aAlphabetConfigPath: str

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

    def enableDecoderWithLM(self, *args, **kwargs):
        """
        Enable decoding using beam scoring with a KenLM language model.

        :param aLMPath: The path to the language model binary file.
        :type aLMPath: str

        :param aTriePath: The path to the trie file build from the same vocabulary as the language model binary.
        :type aTriePath: str

        :param aLMAlpha: The alpha hyperparameter of the CTC decoder. Language Model weight.
        :type aLMAlpha: float

        :param aLMBeta: The beta hyperparameter of the CTC decoder. Word insertion weight.
        :type aLMBeta: float

        :return: Zero on success, non-zero on failure (invalid arguments).
        :type: int
        """
        return deepspeech.impl.EnableDecoderWithLM(self._impl, *args, **kwargs)

    def stt(self, *args, **kwargs):
        """
        Use the DeepSpeech model to perform Speech-To-Text.

        :param aBuffer: A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).
        :type aBuffer: int array

        :param aBufferSize: The number of samples in the audio signal.
        :type aBufferSize: int

        :return: The STT result.
        :type: str
        """
        return deepspeech.impl.SpeechToText(self._impl, *args, **kwargs)

    def sttWithMetadata(self, *args, **kwargs):
        """
        Use the DeepSpeech model to perform Speech-To-Text and output metadata about the results.

        :param aBuffer: A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).
        :type aBuffer: int array

        :param aBufferSize: The number of samples in the audio signal.
        :type aBufferSize: int

        :return: Outputs a struct of individual letters along with their timing information.
        :type: :func:`Metadata`
        """
        return deepspeech.impl.SpeechToTextWithMetadata(self._impl, *args, **kwargs)

    def createStream(self):
        """
        Create a new streaming inference state. The streaming state returned
        by this function can then be passed to :func:`feedAudioContent()` and :func:`finishStream()`.

        :return: Object holding the stream

        :throws: RuntimeError on error
        """
        status, ctx = deepspeech.impl.CreateStream(self._impl)
        if status != 0:
            raise RuntimeError("CreateStream failed with error code {}".format(status))
        return ctx

    # pylint: disable=no-self-use
    def feedAudioContent(self, *args, **kwargs):
        """
        Feed audio samples to an ongoing streaming inference.

        :param aSctx: A streaming state pointer returned by :func:`createStream()`.
        :type aSctx: object

        :param aBuffer: An array of 16-bit, mono raw audio samples at the appropriate sample rate (matching what the model was trained on).
        :type aBuffer: int array

        :param aBufferSize: The number of samples in @p aBuffer.
        :type aBufferSize: int
        """
        deepspeech.impl.FeedAudioContent(*args, **kwargs)

    # pylint: disable=no-self-use
    def intermediateDecode(self, *args, **kwargs):
        """
        Compute the intermediate decoding of an ongoing streaming inference.
        This is an expensive process as the decoder implementation isn't
        currently capable of streaming, so it always starts from the beginning
        of the audio.

        :param aSctx: A streaming state pointer returned by :func:`createStream()`.
        :type aSctx: object

        :return: The STT intermediate result.
        :type: str
        """
        return deepspeech.impl.IntermediateDecode(*args, **kwargs)

    # pylint: disable=no-self-use
    def finishStream(self, *args, **kwargs):
        """
        Signal the end of an audio signal to an ongoing streaming
        inference, returns the STT result over the whole audio signal.

        :param aSctx: A streaming state pointer returned by :func:`createStream()`.
        :type aSctx: object

        :return: The STT result.
        :type: str
        """
        return deepspeech.impl.FinishStream(*args, **kwargs)

    # pylint: disable=no-self-use
    def finishStreamWithMetadata(self, *args, **kwargs):
        """
        Signal the end of an audio signal to an ongoing streaming
        inference, returns per-letter metadata.

        :param aSctx: A streaming state pointer returned by :func:`createStream()`.
        :type aSctx: object

        :return: Outputs a struct of individual letters along with their timing information.
        :type: :func:`Metadata`
        """
        return deepspeech.impl.FinishStreamWithMetadata(*args, **kwargs)

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
        # pylint: disable=unnecessary-pass
        pass

    def timestep(self):
        """
        Position of the character in units of 20ms
        """
        # pylint: disable=unnecessary-pass
        pass

    def start_time(self):
        """
        Position of the character in seconds
        """
        # pylint: disable=unnecessary-pass
        pass


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
        # pylint: disable=unnecessary-pass
        pass

    def num_items(self):
        """
        Size of the list of items

        :return: Size of the list of items
        :type: int
        """
        # pylint: disable=unnecessary-pass
        pass

    def confidence(self):
        """
        Approximated confidence value for this transcription. This is roughly the
        sum of the acoustic model logit values for each timestep/character that
        contributed to the creation of this transcription.
        """
        # pylint: disable=unnecessary-pass
        pass
