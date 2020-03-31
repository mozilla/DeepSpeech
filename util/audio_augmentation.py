from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
from tensorflow.python.ops import gen_audio_ops as contrib_audio
import os
from util.logging import log_info
from util.config import Config


DBFS_COEF = 10.0 / np.log(10.0)

def filename_to_audio(wav_filename):
    r"""Decode `wab_filename` and return the audio

    Args:
        wav_filename: A str, the path of wav file

    Returns:
        A 2-D Tensor with shape [`time-steps`, 1].
    """
    samples = tf.io.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    return decoded.audio

def audio_to_dbfs(audio, sample_rate=16000, chunk_ms=100, reduce_funcs=tf.reduce_mean):
    r"""Separately measure the chunks dbfs of `audio`, then return the statistics values through `reduce_funcs

    Args:
        audio: A 2-D Tensor with shape [`time-steps`, 1].
        sample_rate: An integer, specifying the audio sample rate to determining the chunk size for dbfs measurement.
        chunk_ms: An integer in milliseconds unit, specifying each chunk size for separately measuring dbfs, default is `100ms`
        reduce_funcs: A function or A list of function, specifying the statistics method to chunks, default is tf.reduce_mean

    Returns:
        A float or A list of float, depends on reduce_funcs is function or list of function
    """
    assert chunk_ms % 10 == 0, 'chunk_ms must be a multiple of 10'

    audio_len = tf.shape(audio)[0]
    chunk_len = tf.math.floordiv(sample_rate, tf.math.floordiv(1000, chunk_ms)) # default: 1600
    n_chunks = tf.math.floordiv(audio_len, chunk_len)
    trim_audio_len = tf.multiply(n_chunks, chunk_len)
    audio = audio[:trim_audio_len]
    splits = tf.reshape(audio, shape=[n_chunks, -1])

    squares = tf.square(splits)
    means = tf.reduce_mean(squares, axis=1)

    # the statistics functions must execute before tf.log(), or the gain db would be wrong
    if not isinstance(reduce_funcs, list):
        reduces = reduce_funcs(means)
        return DBFS_COEF * tf.math.log(reduces + 1e-8)

    reduces = [reduce_func(means) for reduce_func in reduce_funcs]
    return [DBFS_COEF * tf.math.log(reduce + 1e-8) for reduce in reduces]


def create_noise_iterator(noise_sources, read_csvs_func):
    r"""Create an iterator to yield audio

    Args:
        noise_dirs_or_files: A list/tuple of str, the collection source of wav filenames.
        read_csvs_func: A function, please specify the `read_csvs()` function from `util/feeding.py`, which is to prevent recursive import error.

    Returns:
        An one shot iterator of audio with 2-D Tensor of shape [`time-step`, 1], use `<iter>.get_next()` to get the Tensor.
    """
    if isinstance(noise_sources, str):
        noise_sources = noise_sources.split(',')

    noise_filenames = tf.convert_to_tensor(list(collect_noise_filenames(noise_sources, read_csvs_func)), dtype=tf.string)
    log_info("Collect {} noise files for mixing audio".format(noise_filenames.shape[0]))

    noise_dataset = (tf.data.Dataset.from_tensor_slices(noise_filenames)
                     .shuffle(min(noise_filenames.shape[0], 102400))
                     .map(filename_to_audio, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                     .prefetch(tfv1.data.experimental.AUTOTUNE)
                     .repeat())
    noise_iterator = tfv1.data.make_one_shot_iterator(noise_dataset)
    return noise_iterator


def collect_noise_filenames(sources, read_csvs_func):
    r"""Collect wav filenames from directories or csv files

    Args:
        dirs_or_files: A list/tuple of str, the collection source of wav filenames.
        read_csvs_func: A function, please specify the `read_csvs()` function from `util/feeding.py`, which is to prevent recursive import error.

    Returns:
        An iterator of str, yield every filename suffix with `.wav` or under `wav_filename` column of DataFrame
    """

    assert isinstance(sources, (list, tuple))

    for source in sources:
        assert os.path.exists(source)
        if os.path.isdir(source):
            for dirpath, _, filenames in os.walk(source):
                for filename in filenames:
                    if filename.endswith('.wav'):
                        yield os.path.join(dirpath, filename)
        elif os.path.isfile(source):
            df = read_csvs_func([source])
            for filename in df['wav_filename']:
                yield filename


def augment_noise(audio,
                  noise_iterator=None,
                  speech_iterator=None,
                  min_n_noises=0,
                  max_n_noises=1,
                  min_n_speakers=0,
                  max_n_speakers=1,
                  min_audio_dbfs=-35.0,
                  max_audio_dbfs=0.0,
                  min_noise_snr_db=3.0,
                  max_noise_snr_db=30.0,
                  min_speech_snr_db=3.0,
                  max_speech_snr_db=30.0,
                  limit_audio_peak_dbfs=7.0,
                  limit_noise_peak_dbfs=3.0,
                  limit_speech_peak_dbfs=7.0,
                  sample_rate=16000):
    r"""Mix audio Tensor with noise Tensor

    If the noise length is shorter than audio, the process will automaticaly repeat the noise file to over audio length,
    The process randomly choose a duration of the noise to complete coverage the audio,
    i.e. the shapes between the choosen duration of noise and audio are equal.

    Args:
        audio: A 2-D Tensor with shape [`time-steps`, 1].
        noise_iterator: A one shot iterator for noise file, the yield item shape is [`time-steps`, 1].
        speech_iterator: A one shot iterator for speech file, the yield item shape is [`time-steps`, 1].
        min_n_noises: A int, min number of the noises per audio mixing
        max_n_noises: A int, 'max number of the noises per audio mixing
        min_n_speakers: A int, min number of the speakers per audio mixing
        max_n_speakers: A int, max number of the speakers per audio mixing
        min_audio_dbfs: A float in dbfs unit, specifying the `minimum` volume of audio during gaining audio.
        max_audio_dbfs: A float in dbfs unit, specifying the `maximum` volume of audio during gaining audio.
        min_noise_snr_db: A float in db unit, specifying the `minimum` signal-to-noise ratio during gaining noise.
        max_noise_snr_db: A float in db unit, specifying the `maximum` signal-to-noise ratio during gaining noise.
        min_speech_snr_db: A float in db unit, specifying the `minimum` signal-to-noise ratio during gaining speech.
        max_speech_snr_db: A float in db unit, specifying the `maximum` signal-to-noise ratio during gaining speech.
        limit_audio_peak_dbfs: A float, specifying the limitation of maximun `audio` dbfs of chunks, the audio volume will not gain over than the specified value.
        limit_noise_peak_dbfs: A float, specifying the limitation of maximun `noise` dbfs of chunks, the noise volume will not gain over than the specified value.
        limit_speech_peak_dbfs: A float, specifying the limitation of maximun `speech` dbfs of chunks, the noise volume will not gain over than the specified value.
        sample_rate: An integer, specifying the audio sample rate to determining the chunk size for dbfs measurement.

    Returns:
        A 2-D Tensor with shape [`time-steps`, 1]. Has the same type and shape as `audio`.
    """

    audio_len = tf.shape(audio)[0]
    audio_mean_dbfs, audio_max_dbfs = audio_to_dbfs(audio, sample_rate, reduce_funcs=[tf.reduce_mean, tf.reduce_max])
    target_audio_dbfs = tfv1.random_uniform([], minval=min_audio_dbfs, maxval=max_audio_dbfs)
    audio_gain_db = target_audio_dbfs - audio_mean_dbfs

    # limit audio peak
    audio_gain_db = tf.minimum(limit_audio_peak_dbfs - audio_max_dbfs, audio_gain_db)
    target_audio_dbfs = audio_mean_dbfs + audio_gain_db
    audio_gain_ratio = tf.math.pow(10.0, audio_gain_db / 20.0)
    mixed_audio = tf.multiply(audio, audio_gain_ratio)


    if noise_iterator:
        n_noise = tfv1.random_uniform([], minval=min_n_noises, maxval=max_n_noises, dtype=tf.int32) if min_n_noises != max_n_noises else min_n_noises
        def mix_noise_func(au):
            noise = noise_iterator.get_next()
            noise, noise_mean_dbfs, noise_max_dbfs = extract_noise(noise, audio_len, sample_rate)
            return mix(au, target_audio_dbfs, noise, noise_mean_dbfs, noise_max_dbfs, min_noise_snr_db, max_noise_snr_db, limit_noise_peak_dbfs)
        mixed_audio = tf.while_loop(lambda _: True, mix_noise_func, [mixed_audio], maximum_iterations=n_noise)

    if speech_iterator:
        n_speakers = tfv1.random_uniform([], minval=min_n_speakers, maxval=max_n_speakers, dtype=tf.int32) if min_n_speakers != max_n_speakers else min_n_speakers
        def mix_speech_func(au):
            speech = speech_iterator.get_next()
            speech, speech_mean_dbfs, speech_max_dbfs = extract_noise(speech, audio_len, sample_rate)
            return mix(au, target_audio_dbfs, speech, speech_mean_dbfs, speech_max_dbfs, min_speech_snr_db, max_speech_snr_db, limit_speech_peak_dbfs)
        mixed_audio = tf.while_loop(lambda _: True, mix_speech_func, [mixed_audio], maximum_iterations=n_speakers)

    mixed_audio = tf.maximum(tf.minimum(mixed_audio, 1.0), -1.0)

    return mixed_audio

def extract_noise(noise, audio_len, sample_rate=16000):
    r"""to prepare the mixable noise file out

    Args:
        noise: A 2-D Tensor with shape [`time-steps`, 1]
        audio_len: A tf.int32 scalar, the audio length
        sample_rate: An integer, specifying the audio sample rate to determining the chunk size for dbfs measurement.
    Returns:
        A 2-D Tensor with shape [`audio_len`, 1].
        A float, the extracted noise mean dbfs
        A float, the extracted noise max dbfs
    """
    noise_len = tf.shape(noise)[0]
    multiply = tf.math.floordiv(audio_len, noise_len) + 1
    noise_tile = tf.tile(noise, [multiply, 1])

    # Now, noise_len must > audio_len
    noise_tile_len = tf.shape(noise_tile)[0]

    mix_decoded_start_point = tfv1.random_uniform([], minval=0, maxval=noise_tile_len-audio_len, dtype=tf.int32)
    mix_decoded_end_point = mix_decoded_start_point + audio_len
    extracted_noise = noise_tile[mix_decoded_start_point:mix_decoded_end_point, :]
    extracted_noise_mean_dbfs, extracted_noise_max_dbfs = audio_to_dbfs(extracted_noise, sample_rate, reduce_funcs=[tf.reduce_mean, tf.reduce_max])
    return extracted_noise, extracted_noise_mean_dbfs, extracted_noise_max_dbfs

def mix(audio, audio_dbfs, noise, noise_mean_dbfs, noise_max_dbfs, min_noise_snr_db, max_noise_snr_db, limit_noise_peak_dbfs):
    r"""The input audio len must equal to noise len

    Returns:
        A 2-D Tensor with shape [`time-steps`, 1]. Has the same type and shape as `audio`.
    """

    # target_snr_db := target_audio_dbfs - target_noise_dbfs
    target_snr_db = tfv1.random_uniform([], minval=min_noise_snr_db, maxval=max_noise_snr_db)

    target_noise_dbfs = audio_dbfs - target_snr_db
    noise_gain_db = target_noise_dbfs - noise_mean_dbfs

    # limit noise peak
    noise_gain_db = tf.minimum(limit_noise_peak_dbfs - noise_max_dbfs, noise_gain_db)
    noise_gain_ratio = tf.math.pow(10.0, noise_gain_db / 20.0)

    audio += tf.multiply(noise, noise_gain_ratio)
    return audio

def gla(spectrogram, n_iter=10):
    r"""Use Griffin-Lim algorithm to reconstruct audio

    Args:
        spectrogram: A 3-D Tensor with shape [1, `time-steps`, `features`].
    Returns:
        A 2-D Tensor with shape [`time-steps`, 1], which is a reconstructed audio from spectrogram.
    """
    frame_length = int(Config.audio_window_samples)
    frame_step = int(Config.audio_step_samples)
    fft_length = 512
    spectrogram = tf.reshape(spectrogram, shape=[1, -1, 257])
    abs_spectrogram = tf.abs(spectrogram)

    def reconstruct_phases(prev_phases):
        xi = tf.complex(abs_spectrogram, 0.0) * prev_phases
        audio = tf.signal.inverse_stft(xi, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
        next_xi = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
        next_phases = tf.math.exp(tf.complex(0.0, tf.angle(next_xi)))
        return next_phases

    rands = tfv1.random_uniform(tf.shape(spectrogram), dtype=tf.float32)
    phases = tf.math.exp(tf.complex(0.0, 2.0 * np.pi * rands))

    reconstructed_phases = tf.while_loop(lambda _: True, reconstruct_phases, [phases], maximum_iterations=n_iter)
    xi = tf.complex(abs_spectrogram, 0.0) * reconstructed_phases
    audio = tf.signal.inverse_stft(xi, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    return tf.transpose(audio)
