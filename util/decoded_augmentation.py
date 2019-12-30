import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.python.ops import gen_audio_ops as contrib_audio
import os


def augment_noise(audio,
                  walk_dirs,
                  change_audio_db_max=0,
                  change_audio_db_min=-10,
                  change_noise_db_max=-25,
                  change_noise_db_min=-50
                  ):
    noise_filenames = []
    for d in walk_dirs:
        for dirpath, _, filenames in os.walk(d):
            for filename in filenames:
                if filename.endswith('.wav'):
                    noise_filenames.append(os.path.join(dirpath, filename))
    print('Collect {} noise filenames for augmentation'.format(len(noise_filenames)))
    noise_filenames = tf.convert_to_tensor(noise_filenames, dtype=tf.string)

    rand_int = tfv1.random_uniform(
        [], dtype=tf.int32, minval=0, maxval=tf.shape(noise_filenames)[0])
    noise_filename = noise_filenames[rand_int]
    noise_samples = tf.io.read_file(noise_filename)
    noise_decoded = contrib_audio.decode_wav(noise_samples, desired_channels=1)
    noise_audio = noise_decoded.audio

    decoded_audio_len = tf.shape(audio)[0]
    noise_decoded_audio_len = tf.shape(noise_audio)[0]

    multiply = tf.math.floordiv(decoded_audio_len, noise_decoded_audio_len) + 1
    noise_audio = tf.tile(noise_audio, [multiply, 1])

    # now noise_decoded_len must > decoded_len
    noise_decoded_audio_len = tf.shape(noise_audio)[0]

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
    extract_noise_decoded = noise_audio[mix_noise_decoded_start_point:mix_noise_decoded_end_point, :]

    right_zeros = tf.zeros(
        shape=[decoded_audio_len - mix_decoded_end_point, 1])

    mixed_noise = tf.concat(
        [left_zeros, extract_noise_decoded, right_zeros], axis=0)

    choosen_audio_db = tfv1.random_uniform(
        [], minval=change_audio_db_min, maxval=change_audio_db_max)
    audio_ratio = tf.math.exp(choosen_audio_db / 10)

    choosen_noise_db = tfv1.random_uniform(
        [], minval=change_noise_db_min, maxval=change_noise_db_max)
    noise_ratio = tf.math.exp(choosen_noise_db / 10)

    return tf.multiply(audio, audio_ratio) + tf.multiply(mixed_noise, noise_ratio)
