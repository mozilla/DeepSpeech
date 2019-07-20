from abc import ABCMeta, abstractmethod

class AugmentorBase(object):
    """Base class for all augmentor
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, mel_fbank):
        """Add different forms of transformations for spectogram
        :param mel_fbanks: mel_fiterbank in log
        :type : tensor (1, τ, v, 1)
        """
        pass
