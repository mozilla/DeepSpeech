import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from util.sparse_image_warp import sparse_image_warp

def augment_freq_time_mask(mel_spectrogram,
                           frequency_masking_para=30,
                           time_masking_para=10,
                           frequency_mask_num=3,
                           time_mask_num=3):
    freq_max = tf.shape(mel_spectrogram)[1]
    time_max = tf.shape(mel_spectrogram)[2]
    # Frequency masking
    for _ in range(frequency_mask_num):
        f = tf.random.uniform(
            shape=(), minval=0, maxval=frequency_masking_para, dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(
            shape=(), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32)
        value_ones_freq_prev = tf.ones(shape=[1, f0, time_max])
        value_zeros_freq = tf.zeros(shape=[1, f, time_max])
        value_ones_freq_next = tf.ones(shape=[1, freq_max-(f0+f), time_max])
        freq_mask = tf.concat(
            [value_ones_freq_prev, value_zeros_freq, value_ones_freq_next], axis=1)
        # mel_spectrogram[:, f0:f0 + f, :] = 0 #can't assign to tensor
        # mel_spectrogram[:, f0:f0 + f, :] = value_zeros_freq #can't assign to tensor
        mel_spectrogram = mel_spectrogram*freq_mask

    # Time masking
    for _ in range(time_mask_num):
        t = tf.random.uniform(shape=(), minval=0,
                              maxval=time_masking_para, dtype=tf.dtypes.int32)
        t0 = tf.random.uniform(
            shape=(), minval=0, maxval=time_max - t, dtype=tf.dtypes.int32)
        value_zeros_time_prev = tf.ones(shape=[1, freq_max, t0])
        value_zeros_time = tf.zeros(shape=[1, freq_max, t])
        value_zeros_time_next = tf.ones(shape=[1, freq_max, time_max-(t0+t)])
        time_mask = tf.concat(
            [value_zeros_time_prev, value_zeros_time, value_zeros_time_next], axis=2)
        # mel_spectrogram[:, :, t0:t0 + t] = 0 #can't assign to tensor
        # mel_spectrogram[:, :, t0:t0 + t] = value_zeros_time #can't assign to tensor
        mel_spectrogram = mel_spectrogram*time_mask

    return mel_spectrogram


def augment_pitch_and_tempo(spectrogram,
                            max_tempo=1.2,
                            max_pitch=1.1,
                            min_pitch=0.95):
    original_shape = tf.shape(spectrogram)
    choosen_pitch = tf.random.uniform(
        shape=(), minval=min_pitch, maxval=max_pitch)
    choosen_tempo = tf.random.uniform(shape=(), minval=1, maxval=max_tempo)
    new_height = tf.cast(
        tf.cast(original_shape[1], tf.float32)*choosen_pitch, tf.int32)
    new_width = tf.cast(
        tf.cast(original_shape[2], tf.float32)/(choosen_tempo), tf.int32)
    spectrogram_aug = tf.image.resize_bilinear(
        tf.expand_dims(spectrogram, -1), [new_height, new_width])
    spectrogram_aug = tf.image.crop_to_bounding_box(spectrogram_aug, offset_height=0, offset_width=0, target_height=tf.minimum(
        original_shape[1], new_height), target_width=tf.shape(spectrogram_aug)[2])
    spectrogram_aug = tf.cond(choosen_pitch < 1,
                              lambda: tf.image.pad_to_bounding_box(spectrogram_aug, offset_height=0, offset_width=0,
                                                                   target_height=original_shape[1], target_width=tf.shape(spectrogram_aug)[2]),
                              lambda: spectrogram_aug)
    return spectrogram_aug[:, :, :, 0]


def augment_speed_up(spectrogram,
                     speed_std=0.1):
    original_shape = tf.shape(spectrogram)
    # abs makes sure the augmention will only speed up
    choosen_speed = tf.math.abs(tf.random.normal(shape=(), stddev=speed_std))
    choosen_speed = 1 + choosen_speed
    new_height = tf.cast(tf.cast(original_shape[1], tf.float32), tf.int32)
    new_width = tf.cast(
        tf.cast(original_shape[2], tf.float32)/(choosen_speed), tf.int32)
    spectrogram_aug = tf.image.resize_bilinear(
        tf.expand_dims(spectrogram, -1), [new_height, new_width])
    return spectrogram_aug[:, :, :, 0]


def augment_dropout(spectrogram,
                    keep_prob=0.95):
    return tf.nn.dropout(spectrogram, rate=1-keep_prob)


def augment_sparse_warp(spectrogram: tf.Tensor, time_warping_para=80, interpolation_order=2, regularization_weight=0.0, num_boundary_points=1):

    if spectrogram.get_shape().ndims == 3:
        spectrogram = tf.expand_dims(spectrogram, -1)
    elif spectrogram.get_shape().ndims == 2:
        spectrogram = tf.expand_dims(tf.expand_dims(spectrogram, 0), -1)
        # spectrogram shape: (1, time steps, freq, 1)

    spec_shape = tf.shape(spectrogram)
    tau, freq_size = spec_shape[1], spec_shape[2] # batch_size must be 1
    time_warping_para = tf.math.minimum(tau, tf.math.floordiv(tau, 2)) # to protect short audio

    mid_tau = tf.math.floordiv(tau, 2)
    mid_freq = tf.math.floordiv(freq_size, 2)
    left_mid_point = [0, mid_freq]
    right_mid_point = [tau, mid_freq]

    time_warping_para = tf.math.minimum(time_warping_para, tf.math.floordiv(tau, 2))
    random_dest_time_point = tfv1.random_uniform(
        [], time_warping_para, tau - time_warping_para, tf.int32)
    # warp
    source_control_point_locations = tf.cast([[
        left_mid_point,
        [mid_tau, mid_freq],
        right_mid_point
    ]], tf.float32)
    dest_control_point_locations = tf.cast([[
        left_mid_point,
        [random_dest_time_point, mid_freq],
        right_mid_point
    ]], tf.float32)

    warped_image, _ = sparse_image_warp(spectrogram,
                                        source_control_point_locations=source_control_point_locations,
                                        dest_control_point_locations=dest_control_point_locations,
                                        interpolation_order=interpolation_order,
                                        regularization_weight=regularization_weight,
                                        num_boundary_points=num_boundary_points)
    return tf.reshape(warped_image, shape=(1, -1, freq_size))
