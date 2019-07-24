# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

from functools import partial

import numpy as np
import pandas
import tensorflow as tf
import datetime

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

from util.config import Config
from util.text import text_to_char_array

def read_csvs(csv_files):
    source_data = None
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        #FIXME: not cross-platform
        csv_dir = os.path.dirname(os.path.abspath(csv))
        file['wav_filename'] = file['wav_filename'].str.replace(r'(^[^/])', lambda m: os.path.join(csv_dir, m.group(1))) # pylint: disable=cell-var-from-loop
        if source_data is None:
            source_data = file
        else:
            source_data = source_data.append(file)
    return source_data


def samples_to_mfccs(samples, sample_rate):
    spectrogram = contrib_audio.audio_spectrogram(samples,
                                                  window_size=Config.audio_window_samples,
                                                  stride=Config.audio_step_samples,
                                                  magnitude_squared=True)
    mfccs = contrib_audio.mfcc(spectrogram, sample_rate, dct_coefficient_count=Config.n_input)
    #FIXME: w.r.t specAugment the audio_spectrogram needs to be changed to mel_frequence bank in log domain
    # mfccs = Config.spectrum_augmentator.transform(mfccs) if Config.spectrum_augmentator else mfccs
    mfccs = tf.reshape(mfccs, [-1, Config.n_input])
    
    return mfccs, tf.shape(mfccs)[0]


def audiofile_to_features(wav_filename):
    samples = tf.io.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    audio = decoded.audio
    if Config.audio_augmentator:
        audio = Config.audio_augmentator.transform(decoded.audio, decoded.sample_rate)

    features, features_len = samples_to_mfccs(audio, decoded.sample_rate)

    return features, features_len


def entry_to_features(wav_filename, transcript):
    # https://bugs.python.org/issue32117
    features, features_len = audiofile_to_features(wav_filename)
    return wav_filename, features, features_len, tf.SparseTensor(*transcript)


def to_sparse_tuple(sequence):
    r"""Creates a sparse representention of ``sequence``.
        Returns a tuple with (indices, values, shape)
    """
    indices = np.asarray(list(zip([0]*len(sequence), range(len(sequence)))), dtype=np.int64)
    shape = np.asarray([1, len(sequence)], dtype=np.int64)
    return indices, sequence, shape


def create_dataset(csvs, batch_size, cache_path=''):
    df = read_csvs(csvs)
    df.sort_values(by='wav_filesize', inplace=True)

    # Convert to character index arrays
    df['transcript'] = df['transcript'].apply(partial(text_to_char_array, alphabet=Config.alphabet))

    def generate_values():
        for _, row in df.iterrows():
            yield row.wav_filename, to_sparse_tuple(row.transcript)

    # Batching a dataset of 2D SparseTensors creates 3D batches, which fail
    # when passed to tf.nn.ctc_loss, so we reshape them to remove the extra
    # dimension here.
    def sparse_reshape(sparse):
        shape = sparse.dense_shape
        return tf.sparse.reshape(sparse, [shape[0], shape[2]])

    def batch_fn(wav_filenames, features, features_len, transcripts):
        features = tf.data.Dataset.zip((features, features_len))
        features = features.padded_batch(batch_size,
                                         padded_shapes=([None, Config.n_input], []))
        transcripts = transcripts.batch(batch_size).map(sparse_reshape)
        wav_filenames = wav_filenames.batch(batch_size)
        return tf.data.Dataset.zip((wav_filenames, features, transcripts))

    num_gpus = len(Config.available_devices)

    dataset = (tf.data.Dataset.from_generator(generate_values,
                                              output_types=(tf.string, (tf.int64, tf.int32, tf.int64)))
                              .map(entry_to_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                              .cache(cache_path)
                              .window(batch_size, drop_remainder=True).flat_map(batch_fn)
                              .prefetch(num_gpus))

    return dataset

def secs_to_hours(secs):
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)
