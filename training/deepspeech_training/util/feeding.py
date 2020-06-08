# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from functools import partial

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import gen_audio_ops as contrib_audio

from .config import Config
from .text import text_to_char_array
from .flags import FLAGS
from .spectrogram_augmentations import augment_freq_time_mask, augment_dropout, augment_pitch_and_tempo, augment_speed_up, augment_sparse_warp
from .audio import read_frames_from_file, vad_split, pcm_to_np, DEFAULT_FORMAT
from .sample_collections import samples_from_sources, augment_samples
from .helpers import remember_exception, MEGABYTE


def samples_to_mfccs(samples, sample_rate, train_phase=False, sample_id=None):
    if train_phase:
        # We need the lambdas to make TensorFlow happy.
        # pylint: disable=unnecessary-lambda
        tf.cond(tf.math.not_equal(sample_rate, FLAGS.audio_sample_rate),
                lambda: tf.print('WARNING: sample rate of sample', sample_id, '(', sample_rate, ') '
                                 'does not match FLAGS.audio_sample_rate. This can lead to incorrect results.'),
                lambda: tf.no_op(),
                name='matching_sample_rate')

    spectrogram = contrib_audio.audio_spectrogram(samples,
                                                  window_size=Config.audio_window_samples,
                                                  stride=Config.audio_step_samples,
                                                  magnitude_squared=True)

    # Data Augmentations
    if train_phase:
        if FLAGS.augmentation_spec_dropout_keeprate < 1:
            spectrogram = augment_dropout(spectrogram,
                                          keep_prob=FLAGS.augmentation_spec_dropout_keeprate)

        # sparse warp must before freq/time masking
        if FLAGS.augmentation_sparse_warp:
            spectrogram = augment_sparse_warp(spectrogram,
                                              time_warping_para=FLAGS.augmentation_sparse_warp_time_warping_para,
                                              interpolation_order=FLAGS.augmentation_sparse_warp_interpolation_order,
                                              regularization_weight=FLAGS.augmentation_sparse_warp_regularization_weight,
                                              num_boundary_points=FLAGS.augmentation_sparse_warp_num_boundary_points,
                                              num_control_points=FLAGS.augmentation_sparse_warp_num_control_points)

        if FLAGS.augmentation_freq_and_time_masking:
            spectrogram = augment_freq_time_mask(spectrogram,
                                                 frequency_masking_para=FLAGS.augmentation_freq_and_time_masking_freq_mask_range,
                                                 time_masking_para=FLAGS.augmentation_freq_and_time_masking_time_mask_range,
                                                 frequency_mask_num=FLAGS.augmentation_freq_and_time_masking_number_freq_masks,
                                                 time_mask_num=FLAGS.augmentation_freq_and_time_masking_number_time_masks)

        if FLAGS.augmentation_pitch_and_tempo_scaling:
            spectrogram = augment_pitch_and_tempo(spectrogram,
                                                  max_tempo=FLAGS.augmentation_pitch_and_tempo_scaling_max_tempo,
                                                  max_pitch=FLAGS.augmentation_pitch_and_tempo_scaling_max_pitch,
                                                  min_pitch=FLAGS.augmentation_pitch_and_tempo_scaling_min_pitch)

        if FLAGS.augmentation_speed_up_std > 0:
            spectrogram = augment_speed_up(spectrogram, speed_std=FLAGS.augmentation_speed_up_std)

    mfccs = contrib_audio.mfcc(spectrogram=spectrogram,
                               sample_rate=sample_rate,
                               dct_coefficient_count=Config.n_input,
                               upper_frequency_limit=FLAGS.audio_sample_rate/2)
    mfccs = tf.reshape(mfccs, [-1, Config.n_input])

    return mfccs, tf.shape(input=mfccs)[0]


def audio_to_features(audio, sample_rate, train_phase=False, sample_id=None):
    features, features_len = samples_to_mfccs(audio, sample_rate, train_phase=train_phase, sample_id=sample_id)

    if train_phase:
        if FLAGS.data_aug_features_multiplicative > 0:
            features = features*tf.random.normal(mean=1, stddev=FLAGS.data_aug_features_multiplicative, shape=tf.shape(features))

        if FLAGS.data_aug_features_additive > 0:
            features = features+tf.random.normal(mean=0.0, stddev=FLAGS.data_aug_features_additive, shape=tf.shape(features))

    return features, features_len


def audiofile_to_features(wav_filename, train_phase=False):
    samples = tf.io.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    return audio_to_features(decoded.audio, decoded.sample_rate, train_phase=train_phase, sample_id=wav_filename)


def entry_to_features(sample_id, audio, sample_rate, transcript, train_phase=False):
    # https://bugs.python.org/issue32117
    features, features_len = audio_to_features(audio, sample_rate, train_phase=train_phase, sample_id=sample_id)
    sparse_transcript = tf.SparseTensor(*transcript)
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
                   repetitions=1,
                   augmentation_specs=None,
                   enable_cache=False,
                   cache_path=None,
                   train_phase=False,
                   exception_box=None,
                   process_ahead=None,
                   buffering=1 * MEGABYTE):
    def generate_values():
        samples = samples_from_sources(sources, buffering=buffering, labeled=True)
        samples = augment_samples(samples,
                                  repetitions=repetitions,
                                  augmentation_specs=augmentation_specs,
                                  buffering=buffering,
                                  process_ahead=2 * batch_size if process_ahead is None else process_ahead)
        for sample in samples:
            transcript = text_to_char_array(sample.transcript, Config.alphabet, context=sample.sample_id)
            transcript = to_sparse_tuple(transcript)
            yield sample.sample_id, sample.audio, sample.audio_format.rate, transcript

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

    process_fn = partial(entry_to_features, train_phase=train_phase)

    dataset = (tf.data.Dataset.from_generator(remember_exception(generate_values, exception_box),
                                              output_types=(tf.string, tf.float32, tf.int32,
                                                            (tf.int64, tf.int32, tf.int64)))
                              .map(process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE))
    if enable_cache:
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
        features, features_len = samples_to_mfccs(samples, audio_format.rate)
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
