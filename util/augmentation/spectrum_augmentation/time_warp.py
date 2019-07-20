from .base import AugmentorBase
import tensorflow as tf

class TimeWarpAugmentor(AugmentorBase):
    """
    TimeWarpAugmentor
    """

    def __init__(self, W=80):
        super(TimeWarpAugmentor, self).__init__()
        self._w = W

    def transform(self, mel_fbank):

        fbank_size = tf.shape(mel_fbank)
        n, v = fbank_size[1], fbank_size[2]

        # Source Points
        init_pt = tf.random_uniform([], self._w, n-self._w, tf.int32)
        src_control_pt_freq = tf.range(v // 2)
        src_control_pt_time = tf.ones_like(src_control_pt_freq) * init_pt
        src_control_pts = tf.stack((src_control_pt_time, src_control_pt_freq), -1)
        src_ctr_pts = tf.to_float(src_control_pts)

        # Destination Points
        w = tf.random_uniform([], -self._w, self._w, tf.int32)
        dest_control_pt_freq = src_control_pt_freq
        dest_control_pt_time = src_control_pt_time + w
        dest_control_pts = tf.stack((dest_control_pt_time, dest_control_pt_freq), -1)
        dest_control_pts = tf.to_float(dest_control_pts)

        # Warping along the time axis
        source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
        dest_control_point_locations = tf.expand_dims(dest_control_pts, 0)  # (1, v//2, 2)

        processd_mel_fbanks, _ = tf.contrib.image.sparse_image_warp(mel_fbanks, source_control_point_locations, dest_control_point_locations)
        return processd_mel_fbanks
