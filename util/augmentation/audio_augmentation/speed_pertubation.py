
from .base import AugmentorBase

import tensorflow as tf
import librosa
import tensorflow.compat.v1 as tfv1

class SpeedPertubation(AugmentorBase):
    def __init__(self, rng, low_speed, high_speed):
        super(SpeedPertubation, self).__init__()
        self._rng = rng
        self._low_speed = low_speed
        self._high_speed = high_speed

    # pylint: disable=fixme
    def _random_time_strech(self, samples):
        """
        Random time stretch pertubation along the time axis
        """
        #FIXME: change to tensorflow complete implementation latter
        samples = samples.flatten().astype('float32')
        speed_range = self._rng.uniform(self._low_speed, self._high_speed)
        samples = librosa.effects.time_stretch(samples, rate=speed_range)
        return samples.reshape([-1, 1])

    def transform(self, audio, sample_rate):
        return tfv1.py_func(self._random_time_strech, [audio], tf.float32)