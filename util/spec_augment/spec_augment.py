import tensorflow as tf
from tensorflow.contrib.image import sparse_image_warp
from tensorflow.python.framework import constant_op
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import numpy as np
import random

# sample_rate=16000
# duration=1000
# frame_length = 256  
# frame_step = 1024 
# num_mel_bins = 40

# desired_samples = int(sample_rate * duration / 1000)
# window_size_samples = int(sr * window_size / 1000)
# window_stride_samples = int(sr * window_stride / 1000)
# difference = (desired_samples - window_size_samples)
# spectrogram_length = 1 + int(difference / window_stride_samples)

def get_log_mel_spectogram(wave_file, sample_rate=16000, \
                            frame_length=1024, frame_step=256, fft_length=51, \
                            lower_edge_hertz=80.0, upper_edge_hertz=7600.0, num_mel_bins=80):
  wav_filename_placeholder = tf.placeholder(tf.string, [])
  wav_loader = io_ops.read_file(wav_filename_placeholder)
  audio_pcm, num_samples = contrib_audio.decode_wav(wav_loader, desired_channels=1)
  
  stfts = tf.signal.stft(tf.transpose(audio_pcm), frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
  spectrograms = tf.abs(stfts)

  num_spectrogram_bins = stfts.shape[-1].value
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                  num_mel_bins, \
                                  num_spectrogram_bins, \
                                  sample_rate, \
                                  lower_edge_hertz, \
                                  upper_edge_hertz
                                )
  mel_spectrograms = tf.tensordot( spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

  log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)

  with tf.Session() as sess:
    spectogram = sess.run(log_mel_spectrograms, feed_dict={wav_filename_placeholder: wave_file})
  
  return spectogram


class Augment:
  """
    Google SpecAugment Implementation
    https://arxiv.org/abs/1904.08779
  """git 
  def __init__(self,v, tau):
    #TODO: Will change this thing, once fix following tensorflow issue
    #https://github.com/tensorflow/tensorflow/issues/28431
    self.input_spectogram = tf.placeholder(tf.float32, [1, None, None])
    self.v = v
    self.tau = tau

  def _time_warp(self, W=5):
    self.spectogram = tf.reshape(self.input_spectogram, [1, self.v, self.tau, 1])
    # self.spectogram.set_shape([spectogram.get_shape()[0], spectogram.get_shape()[1], spectogram.get_shape()[2], spectogram.get_shape()[3]])
    
    # Starting from the center location
    y = tf.math.divide(self.v, 2)
    y = tf.cast(y, tf.int32)

    point_to_warp = tf.random.uniform([1], W, self.tau - W , dtype=tf.int32)
    dist_to_warp = tf.random.uniform([1], -W, W , dtype=tf.int32) # Can either go left or right
    
    src_pts = tf.stack([[y], point_to_warp], 0)
    src_pts = tf.reshape(src_pts, [1, 1, 2])
    src_pts = tf.cast(src_pts, tf.float32, name='source_points')
    
    dest_pts = tf.stack([[y],  tf.add(point_to_warp , dist_to_warp)], 0)
    dest_pts = tf.reshape(dest_pts, [1, 1, 2])
    dest_pts = tf.cast(dest_pts, tf.float32, name='destination_points')

    self.spectogram, _ = sparse_image_warp(self.spectogram,
                                            source_control_point_locations = src_pts,
                                            dest_control_point_locations = dest_pts,
                                            interpolation_order = 2,
                                            regularization_weight = 0,
                                            num_boundary_points = 1
                                          ) 

  def _mask(self, F=30, T=40, num_freq_mask=1, num_time_mask=1, substitution="zero"):
    self.spectogram = tf.squeeze(self.spectogram)
    #TODO: create mask within tensorflow only, to save the moment of data from numpy to tensorflow
    #TODO: add more mask substituition methods like mean, etc
    for _ in range(num_freq_mask):
      mask = np.ones([self.v, self.tau])
      f = np.random.uniform(0.0, F)
      f = int(f)
      f0 = random.randint(0, self.v - f)
      if substitution == "zero":
        mask[f0 : f0 + f, :] = 0
      mask = constant_op.constant(np.float32(mask))
      self.spectogram = tf.multiply(self.spectogram, mask)

    for _ in range(num_time_mask):
      mask = np.ones([self.v, self.tau])
      t = np.random.uniform(0 , T)
      t = int(t)
      t0 = random.randint(0, self.tau - t)
      if substitution == "zero":
        mask[:, t0 : t0 + t] = 0
      mask = constant_op.constant(np.float32(mask))
      self.spectogram = tf.multiply(self.spectogram, mask)

  def execute(self):
    self._time_warp()
    self._mask()
    return (self.input_spectogram, self.spectogram)