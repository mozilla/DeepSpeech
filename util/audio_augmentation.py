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
    decoded_noise_len = tf.shape(noise_audio)[0]

    multiply = tf.math.floordiv(decoded_audio_len, decoded_noise_len) + 1
    noise_audio_tile = tf.tile(noise_audio, [multiply, 1])

    # Now, decoded_noise_len must > decoded_audio_len
    decoded_noise_len = tf.shape(noise_audio_tile)[0]

    mix_decoded_start_point = tfv1.random_uniform(
        [], minval=0, maxval=decoded_noise_len-decoded_audio_len, dtype=tf.int32)
    mix_decoded_end_point = mix_decoded_start_point + decoded_audio_len
    extract_noise_decoded = noise_audio_tile[mix_decoded_start_point:mix_decoded_end_point, :]

    choosen_audio_db = tfv1.random_uniform(
        [], minval=change_audio_db_min, maxval=change_audio_db_max)
    audio_ratio = tf.math.pow(10.0, choosen_audio_db / 10)

    choosen_noise_db = tfv1.random_uniform(
        [], minval=change_noise_db_min, maxval=change_noise_db_max)
    noise_ratio = tf.math.pow(10.0, choosen_noise_db / 10)
    mixed_audio = tf.multiply(audio, audio_ratio) + tf.multiply(extract_noise_decoded, noise_ratio)
    return tf.maximum(tf.minimum(mixed_audio, 1.0), -1.0)
