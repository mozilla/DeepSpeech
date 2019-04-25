import os
import platform

# On Windows, we can't rely on RPATH being set to $ORIGIN/lib/ or on
# @loader_path/lib but we can change the PATH to include the proper directory
# for the dynamic linker
if platform.system().lower() == "windows":
    dslib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')
    os.environ['PATH'] = dslib_path + ';' + os.environ['PATH']

import deepspeech

# rename for backwards compatibility
from deepspeech.impl import PrintVersions as printVersions

class Model(object):
    def __init__(self, *args, **kwargs):
        # make sure the attribute is there if CreateModel fails
        self._impl = None

        status, impl = deepspeech.impl.CreateModel(*args, **kwargs)
        if status != 0:
            raise RuntimeError("CreateModel failed with error code {}".format(status))
        self._impl = impl

    def __del__(self):
        if self._impl:
            deepspeech.impl.DestroyModel(self._impl)
            self._impl = None

    def enableDecoderWithLM(self, *args, **kwargs):
        return deepspeech.impl.EnableDecoderWithLM(self._impl, *args, **kwargs)

    def stt(self, *args, **kwargs):
        return deepspeech.impl.SpeechToText(self._impl, *args, **kwargs)

    def sttWithMetadata(self, *args, **kwargs):
        return deepspeech.impl.SpeechToTextWithMetadata(self._impl, *args, **kwargs)

    def setupStream(self, pre_alloc_frames=150, sample_rate=16000):
        status, ctx = deepspeech.impl.SetupStream(self._impl,
                                                  aPreAllocFrames=pre_alloc_frames,
                                                  aSampleRate=sample_rate)
        if status != 0:
            raise RuntimeError("SetupStream failed with error code {}".format(status))
        return ctx

    def feedAudioContent(self, *args, **kwargs):
        deepspeech.impl.FeedAudioContent(*args, **kwargs)

    def intermediateDecode(self, *args, **kwargs):
        return deepspeech.impl.IntermediateDecode(*args, **kwargs)

    def finishStream(self, *args, **kwargs):
        return deepspeech.impl.FinishStream(*args, **kwargs)

    def finishStreamWithMetadata(self, *args, **kwargs):
        return deepspeech.impl.FinishStreamWithMetadata(*args, **kwargs)
