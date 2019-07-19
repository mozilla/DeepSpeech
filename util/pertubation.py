import tensorflow as tf
import librosa
import numpy as np

def speed_pertub(samples):
    samples = samples.flatten().astype('float32')
    speed_range = np.random.uniform(low=0.8, high=1.4)
    samples = librosa.effects.time_stretch(samples, rate=speed_range)
    return samples.reshape([-1, 1])

def pertub(samples):
    samples = tf.py_func(speed_pertub,[samples], tf.float32)
    return samples