import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.python.ops import gen_audio_ops as contrib_audio
import os

def collect_noise_filenames(walk_dirs):
    assert isinstance(walk_dirs, list)

    for d in walk_dirs:
        for dirpath, _, filenames in os.walk(d):
            for filename in filenames:
                if filename.endswith('.wav'):
                    yield os.path.join(dirpath, filename)

def noise_file_to_audio(noise_file):
    samples = tf.io.read_file(noise_file)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    return decoded.audio

def augment_noise(audio,
                  noise_audio,
                  change_audio_db_max=0,
                  change_audio_db_min=-10,
                  change_noise_db_max=-15,
                  change_noise_db_min=-25):

    decoded_audio_len = tf.shape(audio)[0]
    noise_decoded_audio_len = tf.shape(noise_audio)[0]

    multiply = tf.math.floordiv(decoded_audio_len, noise_decoded_audio_len) + 1
    noise_audio_tile = tf.tile(noise_audio, [multiply, 1])

    # now noise_decoded_len must > decoded_len
    noise_decoded_audio_len = tf.shape(noise_audio_tile)[0]

    mix_decoded_start_end_points = tfv1.random_uniform(
        [2], minval=0, maxval=decoded_audio_len-1, dtype=tf.int32)
    mix_decoded_start_point = tf.math.reduce_min(mix_decoded_start_end_points)
    mix_decoded_end_point = tf.math.reduce_max(
        mix_decoded_start_end_points) + 1
    mix_decoded_width = mix_decoded_end_point - mix_decoded_start_point

    left_zeros = tf.zeros(shape=[mix_decoded_start_point, 1])

    mix_noise_decoded_start_point = tfv1.random_uniform(
        [], minval=0, maxval=noise_decoded_audio_len - mix_decoded_width, dtype=tf.int32)
    mix_noise_decoded_end_point = mix_noise_decoded_start_point + mix_decoded_width
    extract_noise_decoded = noise_audio_tile[mix_noise_decoded_start_point:mix_noise_decoded_end_point, :]

    right_zeros = tf.zeros(
        shape=[decoded_audio_len - mix_decoded_end_point, 1])

    mixed_noise = tf.concat(
        [left_zeros, extract_noise_decoded, right_zeros], axis=0)

    choosen_audio_db = tfv1.random_uniform(
        [], minval=change_audio_db_min, maxval=change_audio_db_max)
    audio_ratio = tf.math.pow(10.0, choosen_audio_db / 10)

    choosen_noise_db = tfv1.random_uniform(
        [], minval=change_noise_db_min, maxval=change_noise_db_max)
    noise_ratio = tf.math.pow(10.0, choosen_noise_db / 10)
    mixed_audio = tf.multiply(audio, audio_ratio) + tf.multiply(mixed_noise, noise_ratio)
    return tf.maximum(tf.minimum(mixed_audio, 1.0), -1.0)
