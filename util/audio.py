import numpy as np
import scipy.io.wavfile as wav

from python_speech_features import fbank

def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    r"""
    Given a WAV audio file at ``audio_filename``, calculates ``numcep`` MFCC features
    at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
    context frames to the left and right of each time step, and returns this data
    in a numpy array.
    """
    # Load wav files
    fs, audio = wav.read(audio_filename)

    # Get MFSC features
    features, energy = fbank(audio, samplerate=fs)
    features = features[:, :numcep]
    features = np.log(features)
    features[:, 0] = np.log(energy)

    orig_inputs = features

    # We only keep every second feature (BiRNN stride = 2)
    orig_inputs = orig_inputs[::2]

    # For each time slice of the training set, we need to copy the context this makes
    # the numcep dimensions vector into a numcep + 2*numcep*numcontext dimensions
    # because of:
    #  - numcep dimensions for the current mfcc feature set
    #  - numcontext*numcep dimensions for each of the past and future (x2) mfcc feature set
    # => so numcep + 2*numcontext*numcep
    train_inputs = np.array([], np.float32)
    train_inputs.resize((orig_inputs.shape[0], numcep + 2*numcep*numcontext))

    # Prepare pre-fix post fix context (TODO: Fill empty_mfcc with MCFF of silence)
    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep))

    # Prepare train_inputs with past and future contexts
    time_slices = range(train_inputs.shape[0])
    context_past_min   = time_slices[0]  + numcontext
    context_future_max = time_slices[-1] - numcontext
    for time_slice in time_slices:
        ### Reminder: array[start:stop:step]
        ### slices from indice |start| up to |stop| (not included), every |step|
        # Pick up to numcontext time slices in the past, and complete with empty
        # mfcc features
        need_empty_past     = max(0, (context_past_min - time_slice))
        empty_source_past   = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past    = orig_inputs[max(0, time_slice - numcontext):time_slice]
        assert(len(empty_source_past) + len(data_source_past) == numcontext)

        # Pick up to numcontext time slices in the future, and complete with empty
        # mfcc features
        need_empty_future   = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future  = orig_inputs[time_slice + 1:time_slice + numcontext + 1]
        assert(len(empty_source_future) + len(data_source_future) == numcontext)

        if need_empty_past:
            past   = np.concatenate((empty_source_past, data_source_past))
        else:
            past   = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future

        past   = np.reshape(past, numcontext*numcep)
        now    = orig_inputs[time_slice]
        future = np.reshape(future, numcontext*numcep)

        train_inputs[time_slice] = np.concatenate((past, now, future))
        assert(len(train_inputs[time_slice]) == numcep + 2*numcep*numcontext)

    # Whiten inputs (TODO: Should we whiten)
    train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)

    # Return results
    return train_inputs
