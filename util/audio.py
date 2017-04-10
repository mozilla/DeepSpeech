from __future__ import absolute_import
import numpy as np
import scipy.io.wavfile as wav

from python_speech_features import mfcc
from six.moves import range

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
    orig_inputs = mfcc(audio, samplerate=fs, numcep=numcep)

    orig_inputs = (orig_inputs - np.mean(orig_inputs))/np.std(orig_inputs)

    return orig_inputs
