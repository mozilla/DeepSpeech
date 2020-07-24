# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from collections import Counter
from functools import partial

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import gen_audio_ops as contrib_audio

from .config import Config
from .text import text_to_char_array
from .flags import FLAGS
from .augmentations import apply_sample_augmentations, apply_graph_augmentations
from .audio import read_frames_from_file, vad_split, pcm_to_np, DEFAULT_FORMAT
from .sample_collections import samples_from_sources
from .helpers import remember_exception, MEGABYTE


def audio_to_features(audio, sample_rate, transcript=None, clock=0.0, train_phase=False, augmentations=None, sample_id=None):
    if train_phase:
        # We need the lambdas to make TensorFlow happy.
        # pylint: disable=unnecessary-lambda
        tf.cond(tf.math.not_equal(sample_rate, FLAGS.audio_sample_rate),
                lambda: tf.print('WARNING: sample rate of sample', sample_id, '(', sample_rate, ') '
                                 'does not match FLAGS.audio_sample_rate. This can lead to incorrect results.'),
                lambda: tf.no_op(),
                name='matching_sample_rate')

    if train_phase and augmentations is not None:
        audio = apply_graph_augmentations('signal', audio, augmentations, transcript=transcript, clock=clock)

    spectrogram = contrib_audio.audio_spectrogram(audio,
                                                  window_size=Config.audio_window_samples,
                                                  stride=Config.audio_step_samples,
                                                  magnitude_squared=True)

    if train_phase and augmentations is not None:
        spectrogram = apply_graph_augmentations('spectrogram', spectrogram, augmentations, transcript=transcript, clock=clock)

    features = contrib_audio.mfcc(spectrogram=spectrogram,
                                  sample_rate=sample_rate,
                                  dct_coefficient_count=Config.n_input,
                                  upper_frequency_limit=FLAGS.audio_sample_rate / 2)
    features = tf.reshape(features, [-1, Config.n_input])

    if train_phase and augmentations is not None:
        features = apply_graph_augmentations('features', features, augmentations, transcript=transcript, clock=clock)

    return features, tf.shape(input=features)[0]


def audiofile_to_features(wav_filename, clock=0.0, train_phase=False, augmentations=None):
    samples = tf.io.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    return audio_to_features(decoded.audio,
                             decoded.sample_rate,
                             clock=clock,
                             train_phase=train_phase,
                             augmentations=augmentations,
                             sample_id=wav_filename)


def entry_to_features(sample_id, audio, sample_rate, transcript, clock, train_phase=False, augmentations=None):
    # https://bugs.python.org/issue32117
    sparse_transcript = tf.SparseTensor(*transcript)
    features, features_len = audio_to_features(audio,
                                               sample_rate,
                                               transcript=sparse_transcript,
                                               clock=clock,
                                               train_phase=train_phase,
                                               augmentations=augmentations,
                                               sample_id=sample_id)
    return sample_id, features, features_len, sparse_transcript


def to_sparse_tuple(sequence):
    r"""Creates a sparse representention of ``sequence``.
        Returns a tuple with (indices, values, shape)
    """
    indices = np.asarray(list(zip([0]*len(sequence), range(len(sequence)))), dtype=np.int64)
    shape = np.asarray([1, len(sequence)], dtype=np.int64)
    return indices, sequence, shape


def create_dataset(sources,
                   batch_size,
                   epochs=1,
                   augmentations=None,
                   cache_path=None,
                   train_phase=False,
                   reverse=False,
                   limit=0,
                   exception_box=None,
                   process_ahead=None,
                   buffering=1 * MEGABYTE):
    epoch_counter = Counter()  # survives restarts of the dataset and its generator

    def generate_values():
        epoch = epoch_counter['epoch']
        if train_phase:
            epoch_counter['epoch'] += 1
        samples = samples_from_sources(sources, buffering=buffering, labeled=True, reverse=reverse)
        num_samples = len(samples)
        if limit > 0:
            num_samples = min(limit, num_samples)
        samples = apply_sample_augmentations(samples,
                                             augmentations,
                                             buffering=buffering,
                                             process_ahead=2 * batch_size if process_ahead is None else process_ahead,
                                             clock=epoch / epochs,
                                             final_clock=(epoch + 1) / epochs)
        for sample_index, sample in enumerate(samples):
            if sample_index >= num_samples:
                break
            clock = (epoch * num_samples + sample_index) / (epochs * num_samples) if train_phase and epochs > 0 else 0.0
            transcript = text_to_char_array(sample.transcript, Config.alphabet, context=sample.sample_id)
            transcript = to_sparse_tuple(transcript)
            yield sample.sample_id, sample.audio, sample.audio_format.rate, transcript, clock

    # Batching a dataset of 2D SparseTensors creates 3D batches, which fail
    # when passed to tf.nn.ctc_loss, so we reshape them to remove the extra
    # dimension here.
    def sparse_reshape(sparse):
        shape = sparse.dense_shape
        return tf.sparse.reshape(sparse, [shape[0], shape[2]])

    def batch_fn(sample_ids, features, features_len, transcripts):
        features = tf.data.Dataset.zip((features, features_len))
        features = features.padded_batch(batch_size, padded_shapes=([None, Config.n_input], []))
        transcripts = transcripts.batch(batch_size).map(sparse_reshape)
        sample_ids = sample_ids.batch(batch_size)
        return tf.data.Dataset.zip((sample_ids, features, transcripts))

    process_fn = partial(entry_to_features, train_phase=train_phase, augmentations=augmentations)

    dataset = (tf.data.Dataset.from_generator(remember_exception(generate_values, exception_box),
                                              output_types=(tf.string, tf.float32, tf.int32,
                                                            (tf.int64, tf.int32, tf.int64), tf.float64))
                              .map(process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE))
    if cache_path:
        dataset = dataset.cache(cache_path)
    dataset = (dataset.window(batch_size, drop_remainder=train_phase).flat_map(batch_fn)
                      .prefetch(len(Config.available_devices)))
    return dataset


def split_audio_file(audio_path,
                     audio_format=DEFAULT_FORMAT,
                     batch_size=1,
                     aggressiveness=3,
                     outlier_duration_ms=10000,
                     outlier_batch_size=1,
                     exception_box=None):
    def generate_values():
        frames = read_frames_from_file(audio_path)
        segments = vad_split(frames, aggressiveness=aggressiveness)
        for segment in segments:
            segment_buffer, time_start, time_end = segment
            samples = pcm_to_np(segment_buffer, audio_format)
            yield time_start, time_end, samples

    def to_mfccs(time_start, time_end, samples):
        features, features_len = audio_to_features(samples, audio_format.rate)
        return time_start, time_end, features, features_len

    def create_batch_set(bs, criteria):
        return (tf.data.Dataset
                .from_generator(remember_exception(generate_values, exception_box),
                                output_types=(tf.int32, tf.int32, tf.float32))
                .map(to_mfccs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .filter(criteria)
                .padded_batch(bs, padded_shapes=([], [], [None, Config.n_input], [])))

    nds = create_batch_set(batch_size,
                           lambda start, end, f, fl: end - start <= int(outlier_duration_ms))
    ods = create_batch_set(outlier_batch_size,
                           lambda start, end, f, fl: end - start > int(outlier_duration_ms))
    dataset = nds.concatenate(ods)
    dataset = dataset.prefetch(len(Config.available_devices))
    return dataset
