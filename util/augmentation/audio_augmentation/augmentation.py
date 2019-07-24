import json
import random
from .speed_pertubation import SpeedPertubation

class AugmentationPipeline(object):
    def __init__(self, augmentation_config, random_seed=0.001):
        self._rng = random.Random(random_seed)
        self._augmentors, self._rates = self._parse_pipeline_from_json(augmentation_config)

    def transform(self, samples, fs):
        """
        Run the augmentation pipeline for audio_augmentation
        as per the config file.

        :param samples:
        :type: tensor
        :param fs:
        :type: tensor
        """
        processed_samples = samples
        for augmentor, rate in zip(self._augmentors, self._rates):
            if self._rng.uniform(0., 1.) < rate:
                processed_samples = augmentor.transform(processed_samples, fs)

        return processed_samples

    def _parse_pipeline_from_json(self, config_json):
        try:
            with open(config_json, 'r') as config_file:
                configs = json.load(config_file)
            augmentors = [self._get_augmentor_by_name(config["type"], config["params"]) for config in configs]
            rates = [config["rate"] for config in configs]
        except json.JSONDecodeError as e:
            raise ValueError("Error parsing the audio augmentation pipeline %s" % str(e))
        return augmentors, rates

    def _get_augmentor_by_name(self, name, params):
        augmentor = None
        if name == "speed_pertub":
            augmentor = SpeedPertubation(self._rng, **params)
        else:
            raise ValueError("Not implemented type of = {%s}." % name)
        return augmentor
