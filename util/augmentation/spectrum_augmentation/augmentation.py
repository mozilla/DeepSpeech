from util.augmentation.base_pipeline import AugmentationPipeline
from .time_warp import TimeWarpAugmentor

class AugmentationPipeline(AugmentationPipeline):
    def __init__(self, augmentation_config, random_seed=0.001):
        super(AugmentationPipeline, self).__init__(augmentation_config, random_seed)

    def transform(self, mel_fbank):
        """
        Run the augmentation pipeline for spectrogram_augmentation
        as per the config file.

        :param mel_fbank: melspectrogram
        :type: tensor (1, τ, v, 1)
        """
        processed_mel_fbank = mel_fbank
        for augmentor, rate in zip(self._augmentors, self._rates):
            if self._rng.uniform(0., 1.) < rate:
                processed_mel_fbank = augmentor.transform(processed_mel_fbank)

        return processed_mel_fbank

    def _get_augmentor_by_name(self, name, params):
        augmentor = None
        if name == "speed_pertub":
            augmentor = TimeWarpAugmentor(self._rng, **params)
        else:
            raise ValueError("Not implemented type of = {%s}." % name)
        return augmentor
