import numpy as np
import scipy.io.wavfile as wav

from python_speech_features import mfcc


def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    r"""
    Given a WAV audio file at ``audio_filename``, calculates ``numcep`` MFCC features
    at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
    context frames to the left and right of each time step, and returns this data
    in a numpy array.
    """
    # Load wav files
    fs, audio = wav.read(audio_filename)

    # Get mfcc coefficients
    features = mfcc(audio, samplerate=fs, numcep=numcep)

    # We only keep every second feature (BiRNN stride = 2)
    features = features[::2]

    # Add empty initial and final contexts
    empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
    features = np.concatenate((empty_context, features, empty_context))

    return features
