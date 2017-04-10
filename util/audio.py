from __future__ import absolute_import
import scipy.io.wavfile as wav
try:
    from deepspeech.utils import audioToInputVector
except ImportError:
    import numpy as np
    from python_speech_features import mfcc
    from six.moves import range

    class DeprecationWarning:
        displayed = False

    def audioToInputVector(audio, fs, numcep, numcontext):
        if DeprecationWarning.displayed is not True:
            DeprecationWarning.displayed = True
            print('------------------------------------------------------------------------')
            print('WARNING: libdeepspeech failed to load, resorting to deprecated code')
            print('         Refer to README.md for instructions on installing libdeepspeech')
            print('------------------------------------------------------------------------')

        # Get mfcc coefficients
        features = mfcc(audio, samplerate=fs, numcep=numcep)

        # Whiten inputs (TODO: Should we whiten?)
        features = (features - np.mean(features))/np.std(features)

        # Return results
        return features


def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    r"""
    Given a WAV audio file at ``audio_filename``, calculates ``numcep`` MFCC features
    at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
    context frames to the left and right of each time step, and returns this data
    in a numpy array.
    """
    # Load wav files
    fs, audio = wav.read(audio_filename)

    return audioToInputVector(audio, fs, numcep, numcontext)
