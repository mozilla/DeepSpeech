"""SpecAugment Implementation for Tensorflow.
Related paper : https://arxiv.org/pdf/1904.08779.pdf
In this paper, show summarized parameters by each open datasets in Tabel 1.
-----------------------------------------
Policy | W  | F  | m_F |  T  |  p  | m_T
-----------------------------------------
None   |  0 |  0 |  -  |  0  |  -  |  -
-----------------------------------------
LB     | 80 | 27 |  1  | 100 | 1.0 | 1
-----------------------------------------
LD     | 80 | 27 |  2  | 100 | 1.0 | 2
-----------------------------------------
SM     | 40 | 15 |  2  |  70 | 0.2 | 2
-----------------------------------------
SS     | 40 | 27 |  2  |  70 | 0.2 | 2
-----------------------------------------
LB : LibriSpeech basic
LD : LibriSpeech double
SM : Switchboard mild
SS : Switchboard strong
"""

import librosa
import librosa.display
import tensorflow as tf
from tensorflow.contrib.image import sparse_image_warp
from tensorflow.python.framework import constant_op
import numpy as np
import random


def spec_augment(mel_spectrogram, time_warping_para=80, frequency_masking_para=27,
                 time_masking_para=100, frequency_mask_num=1, time_mask_num=1):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]

    # Step 1 : Time warping
    # Image warping control point setting.
    center_position = v/2
    random_point = np.random.randint(low=time_warping_para, high=tau - time_warping_para)
    # warping distance chose.
    w = np.random.uniform(low=0, high=time_warping_para)

    control_point_locations = [[center_position, random_point]]
    control_point_locations = constant_op.constant(
        np.float32(np.expand_dims(control_point_locations, 0)))

    control_point_destination = [[center_position, random_point + w]]
    control_point_destination = constant_op.constant(
        np.float32(np.expand_dims(control_point_destination, 0)))

    # mel spectrogram data type convert to tensor constant for sparse_image_warp.
    mel_spectrogram = mel_spectrogram.reshape([1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1])
    mel_spectrogram_op = constant_op.constant(np.float32(mel_spectrogram))

    warped_mel_spectrogram_op, _ = sparse_image_warp(mel_spectrogram_op,
                                                     source_control_point_locations=control_point_locations,
                                                     dest_control_point_locations=control_point_destination,
                                                     interpolation_order=2,
                                                     regularization_weight=0,
                                                     num_boundary_points=1
                                                     )

    # Change warp result's data type to numpy array for masking step.
    with tf.Session() as sess:
        warped_mel_spectrogram = sess.run(warped_mel_spectrogram_op)

    warped_mel_spectrogram = warped_mel_spectrogram.reshape([warped_mel_spectrogram.shape[1],
                                                             warped_mel_spectrogram.shape[2]])

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v - f)
        warped_mel_spectrogram[f0:f0 + f, :] = 0

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        warped_mel_spectrogram[:, t0:t0 + t] = 0

    return warped_mel_spectrogram