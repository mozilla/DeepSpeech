from __future__ import absolute_import, print_function

import scipy.io.wavfile as wav
import sys
import warnings

class DeepSpeechDeprecationWarning(DeprecationWarning):
    pass

warnings.simplefilter('once', category=DeepSpeechDeprecationWarning)

try:
    from deepspeech import audioToInputVector
except ImportError:
    warnings.warn('DeepSpeech Python bindings could not be imported, resorting to slower code to compute audio features. '
                  'Refer to README.md for instructions on how to install (or build) the DeepSpeech Python bindings.',
                  category=DeepSpeechDeprecationWarning)

    import numpy as np
    from python_speech_features import mfcc
    from six.moves import range

    def audioToInputVector(audio, fs, numcep, numcontext):
        # Get mfcc coefficients
        features = mfcc(audio, samplerate=fs, numcep=numcep)

        # We only keep every second feature (BiRNN stride = 2)
        features = features[::2]

        # One stride per time step in the input
        num_strides = len(features)

        # Add empty initial and final contexts
        empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
        features = np.concatenate((empty_context, features, empty_context))

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future)
        window_size = 2*numcontext+1
        train_inputs = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, numcep),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False)

        # Flatten the second and third dimensions
        train_inputs = np.reshape(train_inputs, [num_strides, -1])

        # Return results
        return train_inputs


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
