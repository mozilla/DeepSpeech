from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
from tensorflow.python.ops import gen_audio_ops as contrib_audio
import os
from util.logging import log_info

DBFS_COEF = 20.0 / np.log(10.0)


def get_dbfs(wav_filename):
    samples = tf.io.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    rms = tf.sqrt(tf.reduce_mean(tf.square(decoded.audio)))
    dbfs = DBFS_COEF * tf.math.log(rms)
    return dbfs


def create_noise_iterator(noise_dirs):
    """noise_dirs: `str` or `list`"""
    if isinstance(noise_dirs, str):
        noise_dirs = noise_dirs.split(',')

    noise_filenames = tf.convert_to_tensor(
        list(collect_noise_filenames(noise_dirs)),
        dtype=tf.string)
    log_info("Collect {} noise files for mixing audio".format(
        noise_filenames.shape[0]))

    def extract_dbfs(wav_filename):
        return wav_filename, get_dbfs(wav_filename)
    noise_dataset = (tf.data.Dataset.from_tensor_slices(noise_filenames)
                     .map(extract_dbfs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                     .cache()
                     .shuffle(min(noise_filenames.shape[0], 102400))
                     .map(noise_file_to_audio, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                     .prefetch(tfv1.data.experimental.AUTOTUNE)
                     .repeat())
    noise_iterator = tfv1.data.make_one_shot_iterator(noise_dataset)
    return noise_iterator


def collect_noise_filenames(walk_dirs):
    assert isinstance(walk_dirs, list)

    for d in walk_dirs:
        for dirpath, _, filenames in os.walk(d):
            for filename in filenames:
                if filename.endswith('.wav'):
                    yield os.path.join(dirpath, filename)


def noise_file_to_audio(noise_file, noise_dbfs):
    samples = tf.io.read_file(noise_file)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    return decoded.audio, noise_dbfs


def augment_noise(audio,
                  audio_dbfs,
                  noise,
                  noise_dbfs,
                  max_audio_gain_db=5,
                  min_audio_gain_db=-10,
                  max_snr_db=30,
                  min_snr_db=5):
    decoded_audio_len = tf.shape(audio)[0]
    decoded_noise_len = tf.shape(noise)[0]

    multiply = tf.math.floordiv(decoded_audio_len, decoded_noise_len) + 1
    noise_audio_tile = tf.tile(noise, [multiply, 1])

    # Now, decoded_noise_len must > decoded_audio_len
    decoded_noise_len = tf.shape(noise_audio_tile)[0]

    mix_decoded_start_point = tfv1.random_uniform([], minval=0, maxval=decoded_noise_len-decoded_audio_len, dtype=tf.int32)
    mix_decoded_end_point = mix_decoded_start_point + decoded_audio_len
    extract_noise_decoded = noise_audio_tile[mix_decoded_start_point:mix_decoded_end_point, :]

    audio_gain_db = tfv1.random_uniform([], minval=min_audio_gain_db, maxval=max_audio_gain_db)
    target_audio_dbfs = audio_dbfs + audio_gain_db
    audio_gain_ratio = tf.math.pow(10.0, audio_gain_db / 10)

    # target_snr_db := target_audio_dbfs - target_noise_dbfs
    target_snr_db = tfv1.random_uniform([], minval=min_snr_db, maxval=max_snr_db)

    target_noise_dbfs = target_audio_dbfs - target_snr_db
    noise_gain_db = target_noise_dbfs - noise_dbfs
    noise_gain_ratio = tf.math.pow(10.0, noise_gain_db / 10)
    mixed_audio = tf.multiply(audio, audio_gain_ratio) + tf.multiply(extract_noise_decoded, noise_gain_ratio)
    mixed_audio = tf.maximum(tf.minimum(mixed_audio, 1.0), -1.0)
    return mixed_audio
