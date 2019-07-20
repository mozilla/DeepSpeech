from abc import ABCMeta, abstractmethod

class AugmentorBase(object):
    """Base class for all augmentor
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, samples, fs):
        """Add different forms of transformations for audio pertubation
        :param samples: audio samples in float32 (pcm / (1 << 15))
        :type: tensor
        """
        pass
