import tensorflow as tf
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


def augment_sparse_warp(spectrogram: tf.Tensor, time_warping_para=80):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    if spectrogram.get_shape().ndims == 2:
        spectrogram = tf.reshape(spectrogram, shape=[1, -1, spectrogram.shape[1], 1])
    elif spectrogram.get_shape().ndims == 3:
        spectrogram = tf.reshape(spectrogram, shape=[spectrogram.shape[0], -1, spectrogram.shape[2], 1])
    assert spectrogram.get_shape().ndims == 4
    fbank_size = tf.shape(spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    # Step 1 : Time warping
    # Image warping control point setting.
    # Source
    # radnom point along the time axis
    pt = tf.random.uniform([], time_warping_para, n -
                           time_warping_para, tf.int32)
    src_ctr_pt_freq = tf.range(tf.floordiv(v, 2))  # control points on freq-axis
    # control points on time-axis
    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt
    src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
    src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

    # Destination
    w = tf.random.uniform([], -time_warping_para,
                          time_warping_para, tf.int32)  # distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
    dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

    # warp
    source_control_point_locations = tf.expand_dims(
        src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = tf.expand_dims(
        dest_ctr_pts, 0)  # (1, v//2, 2)

    print(spectrogram.shape)
    warped_image, _ = sparse_image_warp(spectrogram,
                                        source_control_point_locations,
                                        dest_control_point_locations)
    return warped_image
