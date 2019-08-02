import tensorflow as tf
from util.sparse_image_warp import sparse_image_warp

def augment_sparse_deform(mel_spectrogram,
                          time_warping_para=12,
                          normal_around_warping_std=0.5):
    mel_spectrogram = tf.expand_dims(mel_spectrogram, -1)
    freq_max = tf.shape(mel_spectrogram)[1]
    time_max = tf.shape(mel_spectrogram)[2]
    center_freq = tf.cast(freq_max, tf.float32)/2.0
    random_time_point = tf.random.uniform(shape=(), minval=time_warping_para, maxval=tf.cast(time_max, tf.float32) - time_warping_para)
    chosen_warping = tf.random.uniform(shape=(), minval=0, maxval=time_warping_para)
    #add different warping values to different frequencies
    normal_around_warping = tf.random.normal(mean=chosen_warping, stddev=normal_around_warping_std, shape=(3,))

    control_point_freqs = tf.stack([0.0, center_freq, tf.cast(freq_max, tf.float32)], axis=0)
    control_point_times_src = tf.stack([random_time_point, random_time_point, random_time_point], axis=0)
    control_point_times_dst = control_point_times_src+normal_around_warping

    control_src = tf.expand_dims(tf.stack([control_point_freqs, control_point_times_src], axis=-1), 0)
    control_dst = tf.expand_dims(tf.stack([control_point_freqs, control_point_times_dst], axis=1), 0)
    warped_mel_spectrogram, _ = sparse_image_warp(mel_spectrogram,
                                                  source_control_point_locations=control_src,
                                                  dest_control_point_locations=control_dst,
                                                  interpolation_order=2,
                                                  regularization_weight=0,
                                                  num_boundary_points=1
                                                  )
    warped_mel_spectrogram = warped_mel_spectrogram[:, :, :, 0]
    return warped_mel_spectrogram

def augment_freq_time_mask(mel_spectrogram,
                           frequency_masking_para=30,
                           time_masking_para=10,
                           frequency_mask_num=3,
                           time_mask_num=3):
    freq_max = tf.shape(mel_spectrogram)[1]
    time_max = tf.shape(mel_spectrogram)[2]
    # Frequency masking
    # Testing without loop
    for _ in range(frequency_mask_num):
        f = tf.random.uniform(shape=(), minval=0, maxval=frequency_masking_para, dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(shape=(), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32)
        value_ones_freq_prev = tf.ones(shape=[1, f0, time_max])
        value_zeros_freq = tf.zeros(shape=[1, f, time_max])
        value_ones_freq_next = tf.ones(shape=[1, freq_max-(f0+f), time_max])
        freq_mask = tf.concat([value_ones_freq_prev, value_zeros_freq, value_ones_freq_next], axis=1)
        #mel_spectrogram[:, f0:f0 + f, :] = 0 #can't assign to tensor
        #mel_spectrogram[:, f0:f0 + f, :] = value_zeros_freq #can't assign to tensor
        mel_spectrogram = mel_spectrogram*freq_mask

    # Time masking
    # Testing without loop
    for _ in range(time_mask_num):
        t = tf.random.uniform(shape=(), minval=0, maxval=time_masking_para, dtype=tf.dtypes.int32)
        t0 = tf.random.uniform(shape=(), minval=0, maxval=time_max - t, dtype=tf.dtypes.int32)
        value_zeros_time_prev = tf.ones(shape=[1, freq_max, t0])
        value_zeros_time = tf.zeros(shape=[1, freq_max, t])
        value_zeros_time_next = tf.ones(shape=[1, freq_max, time_max-(t0+t)])
        time_mask = tf.concat([value_zeros_time_prev, value_zeros_time, value_zeros_time_next], axis=2)
        #mel_spectrogram[:, :, t0:t0 + t] = 0 #can't assign to tensor
        #mel_spectrogram[:, :, t0:t0 + t] = value_zeros_time #can't assign to tensor
        mel_spectrogram = mel_spectrogram*time_mask

    return mel_spectrogram

def augment_pitch_and_tempo(spectrogram,
                            max_tempo=1.2,
                            max_pitch=1.1,
                            min_pitch=0.95):
    original_shape = tf.shape(spectrogram)
    choosen_pitch = tf.random.uniform(shape=(), minval=min_pitch, maxval=max_pitch)
    choosen_tempo = tf.random.uniform(shape=(), minval=1, maxval=max_tempo)
    new_height = tf.cast(tf.cast(original_shape[1], tf.float32)*choosen_pitch, tf.int32)
    new_width = tf.cast(tf.cast(original_shape[2], tf.float32)/(choosen_tempo), tf.int32)
    spectrogram_aug = tf.image.resize_bilinear(tf.expand_dims(spectrogram, -1), [new_height, new_width])
    spectrogram_aug = tf.image.crop_to_bounding_box(spectrogram_aug, offset_height=0, offset_width=0, target_height=tf.minimum(original_shape[1],new_height), target_width=tf.shape(spectrogram_aug)[2])
    spectrogram_aug = tf.cond(choosen_pitch < 1,
                              lambda: tf.image.pad_to_bounding_box(spectrogram_aug, offset_height=0, offset_width=0,
                                                                   target_height=original_shape[1], target_width=tf.shape(spectrogram_aug)[2]),
                              lambda: spectrogram_aug)
    return spectrogram_aug[:, :, :, 0]


def augment_speed_up(spectrogram,
                     speed_std=0.1):
    original_shape = tf.shape(spectrogram)
    choosen_speed = tf.math.abs(tf.random.normal(shape=(), stddev=speed_std)) # abs makes sure the augmention will only speed up
    choosen_speed = 1 + choosen_speed
    new_height = tf.cast(tf.cast(original_shape[1], tf.float32), tf.int32)
    new_width = tf.cast(tf.cast(original_shape[2], tf.float32)/(choosen_speed), tf.int32)
    spectrogram_aug = tf.image.resize_bilinear(tf.expand_dims(spectrogram, -1), [new_height, new_width])
    return spectrogram_aug[:, :, :, 0]

def augment_dropout(spectrogram,
                    keep_prob=0.95):
    return tf.nn.dropout(spectrogram, rate=1-keep_prob)
