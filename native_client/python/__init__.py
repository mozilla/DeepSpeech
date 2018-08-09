import deepspeech

# rename for backwards compatibility
from deepspeech.impl import AudioToInputVector as audioToInputVector
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
